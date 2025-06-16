import streamlit as st
import gspread
import google.generativeai as genai
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
import faiss
import numpy as np
import os
import pickle
import requests
import time
from google.auth.exceptions import TransportError
from google.api_core import retry

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_URL = "https://api.sarvam.ai/tts"
CREDENTIALS_PATH = "avid-life-462809-s6-47e59658c599.json"
GOOGLE_SHEET_ID = "1K_C7DbuqhWLIF7M5eE_I9I3dcmOcy8eGOMvdEFG324Q"
FAISS_INDEX_PATH = "main_rag_index"
LANGUAGE_VOICES = {
    "English": "en-US-SamanthaNeural",
    "Hindi": "hi-IN-MadhurNeural",
    "Tamil": "ta-IN-ValluvarNeural"
}

def check_environment():
    required_vars = {
        "SARVAM_API_KEY": SARVAM_API_KEY,
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    return True

def get_embedding(text):
    model = genai.GenerativeModel("embedding-001")
    response = model.embed_content(content=text)
    return np.array(response['embedding'], dtype=np.float32)

def load_google_sheet(max_retries=3, delay=5):
    """Load Google Sheet with retry logic"""
    for attempt in range(max_retries):
        try:
            scope = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
            client = gspread.authorize(creds)
            
            # Add retry decorator for the sheet operation
            @retry.Retry(predicate=retry.if_exception_type(TransportError))
            def get_sheet():
                return client.open_by_key(GOOGLE_SHEET_ID).sheet1
            
            sheet = get_sheet()
            records = sheet.get_all_records()
            
            st.success(f"✅ Connected to sheet: {sheet.title}")
            return sheet, records
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                st.error("❌ Failed to connect to Google Sheets after multiple attempts")
                st.error(f"Error details: {str(e)}")
                # Return empty data instead of raising exception
                return None, []

def load_faiss_index():
    if os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss") and os.path.exists(f"{FAISS_INDEX_PATH}/metadata.pkl"):
        index = faiss.read_index(f"{FAISS_INDEX_PATH}/index.faiss")
        with open(f"{FAISS_INDEX_PATH}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []

def check_existing_story(title):
    index, metadata = load_faiss_index()
    if index is None:
        return None
    query_vector = get_embedding(title).reshape(1, -1)
    D, I = index.search(query_vector, 1)
    if I[0][0] < len(metadata):
        result = metadata[I[0][0]]
        if title.lower() in result["title"].lower():
            return result["story"]
    return None

def save_to_faiss_and_sheet(story, title, theme, age, lines, language, audio_url):
    
    embedding = get_embedding(story).reshape(1, -1)
    index, metadata = load_faiss_index()

    if index is None:
        index = faiss.IndexFlatL2(embedding.shape[1])
        metadata = []
    index.add(embedding)
    metadata.append({
        "title": title,
        "theme": theme,
        "audience": age,
        "lines": lines,
        "story": story,
        "language": language,
        "audio_url": audio_url
    })
    
    
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, f"{FAISS_INDEX_PATH}/index.faiss")
    with open(f"{FAISS_INDEX_PATH}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    
    try:
        sheet, _ = load_google_sheet()
        
        row_data = [
            str(title),
            str(story),
            str(theme) if theme else "",
            str(age) if age else "",
            str(lines),
            str(language),
            str(audio_url) if audio_url else ""
        ]
        sheet.append_row(row_data)
        st.success("✅ Successfully saved to Google Sheets!")
    except Exception as e:
        st.error(f"❌ Failed to save to Google Sheets: {str(e)}")
        raise Exception(f"Google Sheets error: {str(e)}")

def generate_story(prompt, retries=3):
    """Generate story with fallback and retry logic"""
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text.strip()
        except exceptions.ResourceExhausted as e:
            if attempt == retries - 1:
                st.error("📢 API quota exceeded. Please try again later or consider upgrading your API plan.")
                return ("I apologize, but I've reached my story generation limit. "
                       "Please try again in a few minutes or contact support.")
            time.sleep(20)  
        except Exception as e:
            st.error(f"Error generating story: {str(e)}")
            return "Error generating story. Please try again."

def generate_audio(text, language):
    voice_id = LANGUAGE_VOICES.get(language, "en-US-SamanthaNeural")
    headers = {
        "accept": "application/json",
        "x-api-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice": voice_id,
        "output_format": "mp3"
    }
    response = requests.post(SARVAM_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("audio_url")
    return None

def save_story_to_sheet(story, title, theme, age, lines, language, audio_url=None):
    try:
        sheet, _ = load_google_sheet()
        row_data = [
            str(title),
            str(story),
            str(theme) if theme else "",
            str(age) if age else "",
            str(lines),
            str(language),
            str(audio_url) if audio_url else ""
        ]
        sheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"Failed to save to sheet: {str(e)}")
        return False


if not check_environment():
    st.stop()

st.set_page_config(page_title="Story Generator", layout="centered")
st.title("📚 AI Story Generator + RAG Storage (Kid-Safe)")
title = st.text_input("📝 Enter Story Title")
language = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Tamil"])
theme = st.text_input("🎭 Enter Theme (Optional)")
age = st.text_input("👶 Age Group (e.g., 6-8 years)")
lines = st.slider("✍️ Number of lines", 3, 20, 5)

if st.button("🚀 Generate Story") and title:
    sheet, data = load_google_sheet()
    if sheet is None:
        st.error("Unable to connect to Google Sheets. Story will be generated but not saved.")
        # Continue with story generation without sheet operations
    existing_story = check_existing_story(title)
    
    if existing_story:
        st.info("📄 Story already exists in RAG. Displaying saved version:")
        st.write(existing_story)
        st.download_button("⬇️ Download Story (Text)", existing_story, file_name=f"{title}.txt")
    else:
        story_prompt = (
            f"You are a creative children's storyteller. Write a {lines}-line story in {language} titled '{title}' "
            f"for the age group {age}. Theme: '{theme}'. Ensure the story is engaging, moral-based or educational, "
            f"and completely safe for children."
        )
        
        generated_story = generate_story(story_prompt)
        if generated_story:
            st.session_state['current_story'] = generated_story
            st.session_state['story_details'] = {
                'title': title,
                'theme': theme,
                'age': age,
                'lines': lines,
                'language': language
            }
            
            st.success("✅ Story generated successfully!")
            edited_story = st.text_area("📝 Edit your story before saving:", 
                                      value=generated_story, 
                                      height=300,
                                      key='story_editor')
            
            
            st.write("🔊 Generating audio narration...")
            audio_url = generate_audio(edited_story, language)
            if audio_url:
                st.audio(audio_url)
                st.markdown(f"[⬇️ Download Audio]({audio_url})")
            else:
                st.warning("⚠️ Could not generate audio.")
            
          
            if st.button("💾 Save Story"):
                if save_story_to_sheet(
                    edited_story,
                    title,
                    theme,
                    age,
                    lines,
                    language,
                    audio_url
                ):
                    st.success("✅ Story saved to Google Sheet successfully!")
                    
                    save_to_faiss_and_sheet(edited_story, title, theme, age, lines, language, audio_url)
                    
            st.download_button("⬇️ Download Story (Text)", 
                             edited_story, 
                             file_name=f"{title}.txt")
else:
    st.info("👆 Enter a title and click Generate Story.")