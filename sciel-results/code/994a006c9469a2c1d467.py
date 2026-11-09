import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def seed_construct():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
        
    G = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    words = set()
    for mask in range(256):
        bits = [(mask >> i) & 1 for i in range(8)]
        word_vec = (G @ np.array(bits)) % 2
        word = 0
        for i, b in enumerate(word_vec):
            if b:
                word |= (1 << i)
        words.add(word)
        
    # Verify
    min_dist = 17
    wl = list(words)
    for i in range(len(wl)):
        for j in range(i+1, len(wl)):
            d = (wl[i] ^ wl[j]).bit_count()
            if d < min_dist:
                min_dist = d
    if min_dist < 6:
        return seed_construct()
        
    current_words = set(wl)
    
    def check_distance(w, words_set):
        for x in words_set:
            if (w ^ x).bit_count() < 6:
                return False
        return True
        
    while time.time() < deadline:
        added = False
        for _ in range(200):
            if time.time() > deadline: break
            w = random.randint(0, 2**17 - 1)
            if w not in current_words and check_distance(w, current_words):
                current_words.add(w)
                added = True
                break
        if not added:
            # Try swap
            if len(current_words) > 10:
                to_remove = random.choice(list(current_words))
                current_words.remove(to_remove)
                found = False
                for _ in range(100):
                    if time.time() > deadline: break
                    w1 = random.randint(0, 2**17 - 1)
                    if w1 in current_words: continue
                    if not check_distance(w1, current_words): continue
                    for _ in range(50):
                        if time.time() > deadline: break
                        w2 = random.randint(0, 2**17 - 1)
                        if w2 in current_words or w2 == w1: continue
                        if not check_distance(w2, current_words): continue
                        if (w1 ^ w2).bit_count() < 6: continue
                        current_words.add(w1)
                        current_words.add(w2)
                        found = True
                        break
                    if found: break
                if not found:
                    current_words.add(to_remove)
                    
    return list(current_words)
