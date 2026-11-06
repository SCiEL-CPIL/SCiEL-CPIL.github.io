import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s kill (evaluator says 60s but use 58 for safety)
    
    n = 17
    min_dist = 6
    max_size = 340  # Upper bound from Brouwer
    total_words = 1 << n  # 131072
    
    # Precompute popcounts using numpy for speed
    # Create array of all 17-bit numbers
    all_words = np.arange(total_words, dtype=np.uint32)
    
    # Compute popcount for all numbers using vectorized bit manipulation
    # For 17-bit numbers, we can use a lookup table approach or direct computation
    # Let's use a simple vectorized approach
    popcount = np.zeros(total_words, dtype=np.uint8)
    for i in range(17):
        popcount += (all_words >> i) & 1
    
    # Strategy: Start with the known 256-word QR code if we can construct it,
    # otherwise use greedy with smart heuristics
    # 
    # The [18,9,6] extended quadratic residue code shortened by one coordinate gives 256 words.
    # For n=17, the quadratic residues mod 17 are: 1, 4, 9, 16, 8, 2, 15, 13 (8 elements)
    # Actually, let's use a different approach: start with 0 and use a greedy algorithm
    # with a "fewest conflicts" heuristic to maximize the code size.
    
    # First, let's try to construct the known 256-word code using the QR code approach
    # The quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # The generator polynomial for the [17,9,5] QR code... this is complex.
    # 
    # Instead, let's use a greedy approach with random restarts and a conflict-based heuristic.
    
    # Precompute conflict information: for each word, count how many other words are within distance < 6
    # This is expensive, so we'll compute it on the fly during greedy.
    
    best_codewords = [0]
    best_size = 1
    
    # Helper: check if a candidate can be added to the current code
    def can_add(candidate, codewords, popcount, min_dist):
        for cw in codewords:
            if popcount[candidate ^ cw] < min_dist:
                return False
        return True
    
    # Helper: run greedy pass with a given order, using a priority queue-like approach
    # Instead of pure greedy, we can try to add words that have the fewest conflicts
    # with the remaining candidate set, but this is expensive.
    
    # Simpler approach: multiple random restarts with greedy in random order
    # plus a local improvement pass (try to swap out one word for two, etc.)
    
    def run_greedy_pass(order, current_codewords=None, popcount=popcount, min_dist=min_dist, deadline=deadline):
        if current_codewords is None:
            codewords = [0]
        else:
            codewords = current_codewords.copy()
        
        for candidate in order:
            if time.time() > deadline - 2.0:
                break
            valid = True
            for cw in codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                codewords.append(candidate)
        return codewords
    
    # Pass 1: Natural order
    if time.time() < deadline - 3.0:
        natural_order = list(range(1, total_words))
        current_cw = run_greedy_pass(natural_order)
        if len(current_cw) > best_size:
            best_size = len(current_cw)
            best_codewords = current_cw
    
    if best_size >= max_size:
        return best_codewords
    
    # Multiple random restarts
    candidates = list(range(1, total_words))
    random.seed(42)  # For reproducibility
    
    while time.time() < deadline - 3.0:
        random.shuffle(candidates)
        current_cw = run_greedy_pass(candidates)
        
        if len(current_cw) > best_size:
            best_size = len(current_cw)
            best_codewords = current_cw
            
            if best_size >= max_size:
                break
        
        # If we have a decent code, try local improvement
        if best_size > 100 and time.time() < deadline - 5.0:
            # Try to improve by removing one word and adding multiple new ones
            # This is a simple 1-for-k exchange
            improved = False
            cw_set = set(best_codewords)
            
            for idx in range(len(best_codewords)):
                if time.time() > deadline - 3.0:
                    break
                # Try removing word at idx
                removed = best_codewords[idx]
                remaining = [best_codewords[j] for j in range(len(best_codewords)) if j != idx]
                
                # Try to add new words
                added = 0
                for candidate in random.sample(candidates, min(1000, len(candidates))):
                    if time.time() > deadline - 3.0:
                        break
                    valid = True
                    for cw in remaining:
                        if popcount[candidate ^ cw] < min_dist:
                            valid = False
                            break
                    if valid and candidate not in cw_set:
                        remaining.append(candidate)
                        cw_set.add(candidate)
                        added += 1
                        if added >= 2:
                            break
                
                if added > 1:  # Net gain of at least 1
                    best_codewords = remaining
                    best_size = len(remaining)
                    improved = True
                    break
            
            if not improved and time.time() > deadline - 10.0:
                break
    
    return best_codewords
