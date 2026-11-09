import time
import random
import itertools
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _popcount(x):
    return bin(x).count("1")

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _reed_muller_extended():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _generate_all_vectors_up_to_weight(n, max_weight):
    """Generate all n-bit vectors with Hamming weight <= max_weight."""
    vectors = [0]
    for w in range(1, max_weight + 1):
        for positions in itertools.combinations(range(n), w):
            v = 0
            for p in positions:
                v |= (1 << p)
            vectors.append(v)
    return vectors

def _expand_forbidden_set(center, n, max_dist):
    """
    Generate all n-bit vectors within Hamming distance <= max_dist from center.
    Used to build the forbidden set.
    """
    forbidden = set()
    # Weight 0
    forbidden.add(center)
    # Weight 1 to max_dist
    for w in range(1, max_dist + 1):
        for positions in itertools.combinations(range(n), w):
            diff = 0
            for p in positions:
                diff |= (1 << p)
            forbidden.add(center ^ diff)
    return forbidden

def _greedy_pack_with_forbidden_set(base_words, n, min_dist, time_limit, seed=42):
    """
    Greedy packing using a forbidden set for O(1) feasibility checks.
    base_words: initial set of codewords (all pairwise distance >= min_dist)
    Returns the final set of codewords.
    """
    rng = random.Random(seed)
    words = list(base_words)
    
    # Build the forbidden set: all vectors within distance < min_dist of any codeword
    # A candidate is feasible if it is NOT in the forbidden set
    forbidden = set()
    for w in words:
        forbidden |= _expand_forbidden_set(w, n, min_dist - 1)
    
    # Also track words as a set for O(1) membership
    word_set = set(words)
    
    start = time.time()
    best_size = len(words)
    
    # Pre-generate a pool of random candidates to avoid repeated RNG calls
    # We'll generate candidates in batches
    
    # Strategy: iterate through candidates, if feasible, add it
    # To be efficient, we can also try structured candidates (low weight) first
    
    # First, try all low-weight vectors not already in the code
    low_weight_vectors = _generate_all_vectors_up_to_weight(n, 4)
    for v in low_weight_vectors:
        if v not in word_set and v not in forbidden:
            words.append(v)
            word_set.add(v)
            forbidden |= _expand_forbidden_set(v, n, min_dist - 1)
            best_size = len(words)
            if time.time() - start > time_limit:
                break
    
    # Then random search
    if time.time() - start < time_limit:
        attempts = 0
        while time.time() - start < time_limit:
            # Generate a batch of candidates
            batch_size = 100
            for _ in range(batch_size):
                if time.time() - start > time_limit:
                    break
                cand = rng.randint(0, (1 << n) - 1)
                if cand not in word_set and cand not in forbidden:
                    words.append(cand)
                    word_set.add(cand)
                    forbidden |= _expand_forbidden_set(cand, n, min_dist - 1)
                    best_size = len(words)
            attempts += batch_size
            # Periodic check
            if time.time() - start > time_limit:
                break
    
    return words

def _local_search_improvement(words, n, min_dist, time_limit, seed=42):
    """
    Try to improve the code by removing one word and adding two others.
    This is a simple 1-out-2-in local search.
    """
    rng = random.Random(seed)
    words = list(words)
    word_set = set(words)
    forbidden = set()
    for w in words:
        forbidden |= _expand_forbidden_set(w, n, min_dist - 1)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing each word and see if we can add more
        n_words = len(words)
        for i in range(n_words):
            if time.time() - start > time_limit:
                break
            # Remove word i
            removed = words[i]
            # Rebuild forbidden set without word i (expensive, but n is small)
            # Actually, let's just try: remove word i, then try to add 2 new words
            new_words = [w for j, w in enumerate(words) if j != i]
            new_word_set = set(new_words)
            new_forbidden = set()
            for w in new_words:
                new_forbidden |= _expand_forbidden_set(w, n, min_dist - 1)
            
            # Try to add 2 new words
            added = 0
            for _ in range(50):
                if time.time() - start > time_limit:
                    break
                cand = rng.randint(0, (1 << n) - 1)
                if cand not in new_word_set and cand not in new_forbidden:
                    new_words.append(cand)
                    new_word_set.add(cand)
                    new_forbidden |= _expand_forbidden_set(cand, n, min_dist - 1)
                    added += 1
                    if added >= 2:
                        break
            
            if added >= 2:
                words = new_words
                word_set = new_word_set
                forbidden = new_forbidden
                improved = True
                break
    
    return words

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()
    
    # 1. Start with Reed-Muller code (32 words)
    rm_words = _reed_muller_extended()
    if not _check_min_distance(rm_words, MIN_DIST):
        raise ValueError("RM code failed distance check")
    
    # 2. Use greedy packing with forbidden set
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    if time_remaining > 10.0:
        # Allocate time: 60% to greedy packing, 30% to local search, 10% buffer
        greedy_time = time_remaining * 0.6
        local_time = time_remaining * 0.3
        
        best = _greedy_pack_with_forbidden_set(rm_words, N, MIN_DIST, time_limit=greedy_time, seed=42)
        
        # Verify
        if _check_min_distance(best, MIN_DIST):
            # Try local search
            elapsed2 = time.time() - start
            time_remaining2 = DEADLINE - elapsed2
            if time_remaining2 > 5.0:
                improved = _local_search_improvement(best, N, MIN_DIST, time_limit=min(local_time, time_remaining2 - 2.0), seed=123)
                if len(improved) > len(best) and _check_min_distance(improved, MIN_DIST):
                    best = improved
        
        return best
    
    # Fallback: just return RM code
    return rm_words
