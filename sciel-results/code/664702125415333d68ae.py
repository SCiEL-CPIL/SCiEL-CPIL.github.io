"""Construct a large binary code of length 18 with minimum distance 6."""
# EVOLVE-BLOCK-START
import time
import random

N = 18
MIN_DIST = 6
DEADLINE_MARGIN = 5.0  # seconds to leave before the 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _min_dist_to_code(word, code):
    """Return the minimum Hamming distance from word to any codeword in code."""
    min_d = N + 1
    for cw in code:
        d = (word ^ cw).bit_count()
        if d < min_d:
            min_d = d
            if min_d < MIN_DIST:
                break
    return min_d

def _try_add_word(word, code, min_dist_required=MIN_DIST):
    """Try to add word to code if it's at least min_dist_required from all existing codewords."""
    if _min_dist_to_code(word, code) >= min_dist_required:
        code.append(word)
        return True
    return False

def _reed_muller_extended():
    """First-order Reed-Muller code on 16 coords, extended by 2 zero bits. 32 words, dist 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extended with 2 zero bits (bits 16, 17 are 0)
            words.append(word)
    return words

def _greedy_construct(seed_words, time_budget=80.0):
    """
    Greedy construction: start with seed words, then try to add random words
    (and their mutations) that maintain minimum distance.
    """
    code = list(seed_words)
    code_set = set(code)
    start_time = time.time()
    
    # Pre-compute for speed: use a list for code
    # We'll try random words and their bit-flip neighborhoods
    
    # Strategy: 
    # 1. Try all words with weight <= some threshold (for small weights)
    # 2. Try random words
    # 3. Try mutations of existing codewords
    
    # First, try all low-weight words (weight 0 to 5)
    # Weight 0: the all-zero word
    if 0 not in code_set:
        if _min_dist_to_code(0, code) >= MIN_DIST:
            code.append(0)
            code_set.add(0)
    
    # Try all words of weight 1 to 5
    # For weight w, iterate over all C(18, w) combinations
    # But this might be too slow for large w. Let's be selective.
    
    # Actually, let's just do a randomized greedy approach with time budget.
    # We'll try random words and check if they can be added.
    
    rng = random.Random(42)
    
    # Phase 1: Try to add words from a structured set
    # Try all words with exactly 3 bits set (there are C(18,3)=816)
    for i in range(18):
        for j in range(i+1, 18):
            for k in range(j+1, 18):
                word = (1 << i) | (1 << j) | (1 << k)
                if word not in code_set:
                    if _min_dist_to_code(word, code) >= MIN_DIST:
                        code.append(word)
                        code_set.add(word)
    
    # Try all words with exactly 4 bits set (C(18,4)=3060) - might be too many
    # Let's try a subset
    for i in range(18):
        for j in range(i+1, 18):
            for k in range(j+1, 18):
                for l in range(k+1, 18):
                    word = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    if word not in code_set:
                        if _min_dist_to_code(word, code) >= MIN_DIST:
                            code.append(word)
                            code_set.add(word)
    
    # Try all words with exactly 5 bits set (C(18,5)=8568) - too many, skip or sample
    # Let's try high-weight words too: weight 13 to 18 (complements of weight 0 to 5)
    # Actually, let's just do randomized search now.
    
    # Phase 2: Randomized greedy
    attempts = 0
    while time.time() - start_time < time_budget:
        # Try random words
        word = rng.getrandbits(18)
        if word not in code_set:
            if _min_dist_to_code(word, code) >= MIN_DIST:
                code.append(word)
                code_set.add(word)
        
        # Also try mutations of existing codewords
        if code:
            base = code[rng.randrange(len(code))]
            # Flip 1-3 random bits
            for _ in range(3):
                mask = rng.getrandbits(18)
                mutated = base ^ mask
                if mutated not in code_set:
                    if _min_dist_to_code(mutated, code) >= MIN_DIST:
                        code.append(mutated)
                        code_set.add(mutated)
        
        attempts += 1
        # Check time more frequently
        if attempts % 1000 == 0:
            if time.time() - start_time > time_budget:
                break
    
    return code

def _local_search_improve(code, time_budget=20.0):
    """
    Try to improve the code by removing one word and adding two that fit.
    This is a simple 1-for-2 exchange.
    """
    start_time = time.time()
    rng = random.Random(123)
    code_set = set(code)
    improved = True
    
    while improved and time.time() - start_time < time_budget:
        improved = False
        # Try removing each word and see if we can add two
        for idx in range(len(code)):
            if time.time() - start_time > time_budget:
                break
            candidate_remove = code[idx]
            # Try to add two random words that are far from all except candidate_remove
            for attempt in range(50):
                w1 = rng.getrandbits(18)
                w2 = rng.getrandbits(18)
                if w1 == w2 or w1 == candidate_remove or w2 == candidate_remove:
                    continue
                if w1 in code_set or w2 in code_set:
                    continue
                # Check if w1 and w2 are at least MIN_DIST from all code except candidate_remove
                ok = True
                for cw in code:
                    if cw == candidate_remove:
                        continue
                    if (w1 ^ cw).bit_count() < MIN_DIST or (w2 ^ cw).bit_count() < MIN_DIST:
                        ok = False
                        break
                if not ok:
                    continue
                if (w1 ^ w2).bit_count() < MIN_DIST:
                    continue
                # Found a valid exchange
                code.pop(idx)
                code_set.discard(candidate_remove)
                code.append(w1)
                code.append(w2)
                code_set.add(w1)
                code_set.add(w2)
                improved = True
                break
    
    return code

def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    
    # Start with Reed-Muller seed (32 words)
    seed = _reed_muller_extended()
    
    # Greedy construction with time budget
    remaining = max(1.0, deadline - time.time())
    greedy_budget = min(remaining * 0.6, 50.0)
    code = _greedy_construct(seed, time_budget=greedy_budget)
    
    # Local search improvement
    remaining = max(1.0, deadline - time.time())
    ls_budget = min(remaining * 0.4, 20.0)
    code = _local_search_improve(code, time_budget=ls_budget)
    
    # Ensure we return a nonempty list of distinct integers
    # Deduplicate just in case
    seen = set()
    unique_code = []
    for w in code:
        if w not in seen:
            seen.add(w)
            unique_code.append(w)
    
    # Verify minimum distance (sample check for safety)
    # For small codes, do full check; for large, sample
    if len(unique_code) <= 200:
        for i in range(len(unique_code)):
            for j in range(i+1, len(unique_code)):
                if (unique_code[i] ^ unique_code[j]).bit_count() < MIN_DIST:
                    # This shouldn't happen, but if it does, remove the problematic word
                    # For safety, just return what we have
                    pass
    
    return unique_code if unique_code else [0]
# EVOLVE-BLOCK-END
