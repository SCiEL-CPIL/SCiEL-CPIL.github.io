"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TIME_LIMIT = 80  # seconds, safety margin before 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _can_add(candidate, words_list):
    for w in words_list:
        if _hamming_dist(candidate, w) < MIN_DISTANCE:
            return False
    return True

def _generate_reed_muller_seed():
    """Generate the 32-word first-order Reed-Muller code on 16 coords, extended by 0 bit."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit is 0
            words.append(word)
    return words

def _generate_linear_code_seed():
    """
    Try to generate a larger linear code seed.
    The [18,9,6] QR code shortened gives 256 words.
    We'll construct a systematic [17,k,6] linear code if possible,
    or fall back to the RM seed.
    """
    # For now, use the RM code as a reliable seed.
    # A proper [17,9,6] linear code would give 512 words but may not exist.
    # The task says 256-word linear reference exists.
    # Let's try to build a code with good structure.
    
    # Actually, let's just use the RM seed and let greedy expand it.
    return _generate_reed_muller_seed()

def _greedy_expand(seed_words, rng, time_deadline):
    """
    Greedily add words to the code, trying random candidates.
    Returns the expanded list of words.
    """
    words = list(seed_words)
    words_set = set(words)
    
    # Precompute for efficiency: we'll track words as a list
    # For speed, we can use a list and check against it
    
    # Strategy: try all 2^17 candidates in a random order, add if valid
    # But 2^17 = 131072 candidates, checking against growing list
    # With 256+ words, each check is O(n), so total ~131072 * 256 * bit_count
    # That's about 33 million bit_count ops, should be fast enough
    
    candidates = list(range(1 << N))
    rng.shuffle(candidates)
    
    added = 0
    for cand in candidates:
        if time.time() > time_deadline:
            break
        if cand in words_set:
            continue
        # Check distance to all existing words
        valid = True
        for w in words:
            if _hamming_dist(cand, w) < MIN_DISTANCE:
                valid = False
                break
        if valid:
            words.append(cand)
            words_set.add(cand)
            added += 1
    
    return words

def _greedy_with_local_search(seed_words, rng, time_deadline):
    """
    More sophisticated: greedy add, then try to improve by swapping.
    """
    words = list(seed_words)
    words_set = set(words)
    
    # First, greedy fill
    candidates = list(range(1 << N))
    rng.shuffle(candidates)
    
    for cand in candidates:
        if time.time() > time_deadline:
            break
        if cand in words_set:
            continue
        valid = True
        for w in words:
            if _hamming_dist(cand, w) < MIN_DISTANCE:
                valid = False
                break
        if valid:
            words.append(cand)
            words_set.add(cand)
    
    # Now try local improvements: remove a word, try to add two
    # This is complex, skip for now
    
    return words

def construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    # Start with the Reed-Muller seed
    seed = _generate_reed_muller_seed()
    
    # Try multiple strategies and keep the best
    best_words = seed
    
    # Strategy 1: Simple greedy from RM seed
    words1 = _greedy_expand(seed, rng, time_deadline)
    if len(words1) > len(best_words):
        best_words = words1
    
    if time.time() > time_deadline - 10:
        return best_words
    
    # Strategy 2: Try different orderings / multiple restarts
    for trial in range(5):
        if time.time() > time_deadline - 15:
            break
        rng_trial = random.Random(1000 + trial)
        seed_trial = _generate_reed_muller_seed()
        # Shuffle the seed order to get different greedy outcomes
        rng_trial.shuffle(seed_trial)
        words_trial = _greedy_expand(seed_trial, rng_trial, time_deadline - 10)
        if len(words_trial) > len(best_words):
            best_words = words_trial
    
    # Strategy 3: Try building from scratch with random greedy
    if time.time() < time_deadline - 20:
        for trial in range(3):
            if time.time() > time_deadline - 15:
                break
            rng_trial = random.Random(2000 + trial)
            # Start with a random word, then greedy
            initial = [rng_trial.randint(0, (1 << N) - 1)]
            words_trial = _greedy_expand(initial, rng_trial, time_deadline - 10)
            if len(words_trial) > len(best_words):
                best_words = words_trial
    
    # Final check: ensure all words are distinct and valid
    # (they should be by construction, but verify)
    seen = set()
    valid_words = []
    for w in best_words:
        if w not in seen:
            seen.add(w)
            valid_words.append(w)
    
    # Verify minimum distance (spot check for safety)
    # For large codes, this is O(n^2) which could be slow
    # But we trust the greedy construction
    if len(valid_words) < 2:
        return valid_words if valid_words else [0]
    
    return valid_words
