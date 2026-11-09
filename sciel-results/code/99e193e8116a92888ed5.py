"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

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

def _precompute_deltas():
    """Precompute all vectors of weight <= 5 (the exclusion zone radius)."""
    universe_size = 1 << N
    deltas = []
    for d in range(universe_size):
        if d.bit_count() <= 5:
            deltas.append(d)
    return np.array(deltas, dtype=np.uint32)

def _greedy_expand_structured(seed_words, rng, time_deadline, deltas):
    """
    Structured greedy expansion using a forbidden-zone array.
    Each added word marks all words within distance 5 as forbidden.
    """
    words = list(seed_words)
    universe_size = 1 << N
    forbidden = np.zeros(universe_size, dtype=bool)
    
    # Initialize forbidden zone for seed words
    for w in words:
        neighbors = (w ^ deltas).astype(np.uint32)
        forbidden[neighbors] = True
    
    # Random order for candidate selection
    indices = np.arange(universe_size)
    rng.shuffle(indices)
    
    for idx in indices:
        if time.time() > time_deadline - 2:
            break
        if forbidden[idx]:
            continue
        w = int(idx)
        words.append(w)
        neighbors = (w ^ deltas).astype(np.uint32)
        forbidden[neighbors] = True
    
    return words

def _local_search_swap(words, rng, time_deadline, deltas):
    """
    Local search: try to remove one word and add two others.
    Rebuilds the forbidden zone from scratch for each swap attempt.
    """
    words = list(words)
    universe_size = 1 << N
    best_words = words
    best_size = len(words)
    
    def rebuild_forbidden(word_list):
        forbidden = np.zeros(universe_size, dtype=bool)
        for w in word_list:
            neighbors = (w ^ deltas).astype(np.uint32)
            forbidden[neighbors] = True
        return forbidden
    
    # Try a limited number of swap attempts
    max_swaps = 20
    for _ in range(max_swaps):
        if time.time() > time_deadline - 2:
            break
        if len(best_words) < 2:
            break
        
        # Pick a random word to remove
        idx = rng.randint(0, len(best_words) - 1)
        removed = best_words[idx]
        temp_words = best_words[:idx] + best_words[idx+1:]
        
        # Rebuild forbidden zone without the removed word
        forbidden = rebuild_forbidden(temp_words)
        
        # Try to find 2 new words
        added = []
        indices = np.arange(universe_size)
        rng.shuffle(indices)
        
        for idx_cand in indices:
            if len(added) == 2:
                break
            if forbidden[idx_cand]:
                continue
            w = int(idx_cand)
            temp_words.append(w)
            added.append(w)
            neighbors = (w ^ deltas).astype(np.uint32)
            forbidden[neighbors] = True
        
        if len(added) == 2:
            if len(temp_words) > best_size:
                best_words = temp_words
                best_size = len(temp_words)
    
    return best_words

def construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    rng = random.Random(42)
    
    # Start with the Reed-Muller seed (32 words)
    seed = _generate_reed_muller_seed()
    
    # Precompute deltas once
    deltas = _precompute_deltas()
    
    best_words = seed
    best_size = len(seed)
    
    # Strategy: Structured Greedy Expansion with multiple random orderings
    num_trials = 15
    for trial in range(num_trials):
        if time.time() > time_deadline - 10:
            break
        
        rng_trial = random.Random(1000 + trial)
        words_trial = _greedy_expand_structured(seed, rng_trial, time_deadline - 5, deltas)
        
        if len(words_trial) > best_size:
            best_words = words_trial
            best_size = len(words_trial)
        
        # Early exit if we've already exceeded the known linear code size
        if best_size > 256:
            break
    
    # Strategy: Local Search Improvement
    if time.time() < time_deadline - 10 and len(best_words) > 50:
        rng_ls = random.Random(2000)
        improved_words = _local_search_swap(best_words, rng_ls, time_deadline - 5, deltas)
        if len(improved_words) > best_size:
            best_words = improved_words
            best_size = len(improved_words)
    
    # Final verification and cleanup
    seen = set()
    valid_words = []
    for w in best_words:
        if w not in seen:
            seen.add(w)
            valid_words.append(w)
    
    if len(valid_words) < 2:
        return valid_words if valid_words else [0]
    
    return valid_words
