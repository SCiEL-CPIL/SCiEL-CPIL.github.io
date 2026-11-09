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

def _greedy_expand_vectorized(seed_words, rng, time_deadline):
    """
    Greedily add words to the code using vectorized distance checks.
    Returns the expanded list of words.
    """
    words = list(seed_words)
    words_set = set(words)
    
    # Precompute all candidates
    candidates = np.array(list(range(1 << N)), dtype=np.uint32)
    
    # Shuffle candidates
    rng.shuffle(candidates)
    
    # We will process in chunks to check time
    chunk_size = 10000
    n_candidates = len(candidates)
    
    # Convert current words to numpy array for vectorized ops
    # We need to check distance of each candidate against all current words.
    # To be efficient, we can maintain a list of words and check new candidates against it.
    # Vectorized approach: 
    # 1. Take a chunk of candidates.
    # 2. For each candidate, compute distance to all existing words.
    # 3. If min distance >= MIN_DISTANCE, add it.
    
    # However, doing this for every candidate in a chunk against a growing list is O(N_chunk * N_words).
    # With N_words growing to ~100+, and N_chunk=10000, that's 1M distance checks per chunk.
    # 1M * bit_count is fast in numpy? 
    # np.bitwise_xor(candidate_chunk[:, None], words_arr[None, :]) -> (10000, N_words) matrix
    # Then .bit_count() is not directly a numpy method for integers in older versions, 
    # but we can use bin(x).count('1') or precomputed popcount?
    # Actually, numpy doesn't have a direct bit_count for integers in all versions.
    # We can use: np.sum((arr != 0), axis=...) after shifting? No, that's complex.
    # Let's stick to a hybrid: 
    # If N_words is small, use Python loop with vectorized candidate check?
    # Or just use the standard greedy but with a better ordering?
    
    # The current program's greedy is O(N_candidates * N_words).
    # 131k * 100 = 13M operations. Python bit_count is fast.
    # The bottleneck in the current program might be the Python loop overhead.
    
    # Let's try a different strategy: 
    # 1. Start with RM seed.
    # 2. Use a priority queue or just iterate through candidates in a smart order.
    # 3. Instead of random shuffle, try to add candidates that are "far" from existing ones.
    
    # For now, let's optimize the existing greedy by using a set for O(1) lookup 
    # and checking distances in a vectorized way if possible, or just optimized Python.
    
    # Let's try a "local search" approach on top of greedy.
    # 1. Greedy fill.
    # 2. Try to swap: remove word i, try to add 2 new words that fit.
    
    # First, let's just do a fast greedy.
    # To speed up, we can pre-compute popcounts for all 2^17 numbers?
    # popcount[i] = i.bit_count()
    # Then dist(a, b) = popcount[a ^ b]
    
    popcount = [i.bit_count() for i in range(1 << N)]
    
    def fast_dist(a, b):
        return popcount[a ^ b]
    
    # Greedy Pass 1
    for i in range(n_candidates):
        if time.time() > time_deadline - 5:
            break
        cand = candidates[i]
        if cand in words_set:
            continue
        
        valid = True
        for w in words:
            if fast_dist(cand, w) < MIN_DISTANCE:
                valid = False
                break
        
        if valid:
            words.append(cand)
            words_set.add(cand)
    
    return words

def _local_search_improve(words, rng, time_deadline):
    """
    Try to improve the code by removing one word and adding two others.
    """
    words = list(words)
    words_set = set(words)
    popcount = [i.bit_count() for i in range(1 << N)]
    
    def fast_dist(a, b):
        return popcount[a ^ b]
    
    improved = True
    while improved and time.time() < time_deadline - 5:
        improved = False
        # Shuffle indices to try different removals
        indices = list(range(len(words)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > time_deadline - 5:
                break
            
            removed_word = words[idx]
            
            # Temporarily remove
            words.pop(idx)
            words_set.remove(removed_word)
            
            # Try to add two new words
            added = []
            # Try random candidates
            for _ in range(100): # Limit attempts per removal
                if time.time() > time_deadline - 5:
                    break
                cand = rng.randint(0, (1 << N) - 1)
                if cand in words_set or cand == removed_word:
                    continue
                
                # Check distance to all current words
                valid = True
                for w in words:
                    if fast_dist(cand, w) < MIN_DISTANCE:
                        valid = False
                        break
                
                if valid:
                    words.append(cand)
                    words_set.add(cand)
                    added.append(cand)
                    if len(added) == 2:
                        break
            
            if len(added) >= 2:
                improved = True
            elif len(added) == 1:
                # We added 1, but removed 1, so size unchanged. 
                # Maybe keep the new one if it's "better"? 
                # For now, just revert if we didn't gain size.
                # But if we added 1, we are at same size. 
                # Let's only accept if size increased.
                # So if len(added) < 2, revert.
                words.pop()
                words_set.remove(added[0])
            else:
                # Revert
                words.insert(idx, removed_word)
                words_set.add(removed_word)
    
    return words

def construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    rng = random.Random(42)
    
    # Start with the Reed-Muller seed
    seed = _generate_reed_muller_seed()
    
    best_words = seed
    best_size = len(seed)
    
    # Strategy 1: Greedy expand with multiple random orderings
    for trial in range(5):
        if time.time() > time_deadline - 10:
            break
        
        rng_trial = random.Random(1000 + trial)
        words_trial = _greedy_expand_vectorized(seed, rng_trial, time_deadline - 5)
        
        if len(words_trial) > best_size:
            best_words = words_trial
            best_size = len(words_trial)
    
    # Strategy 2: Local search improvement on the best greedy result
    if time.time() < time_deadline - 10:
        rng_ls = random.Random(2000)
        improved_words = _local_search_improve(best_words, rng_ls, time_deadline - 5)
        if len(improved_words) > best_size:
            best_words = improved_words
            best_size = len(best_words)
    
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
