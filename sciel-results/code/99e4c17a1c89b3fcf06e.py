import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Seed: First-order Reed-Muller code on 16 coordinates, 17th bit = 0.
    # This gives 32 words of length 17 with distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    code = set(words)
    
    # Precompute all candidates as numpy array
    all_candidates = np.arange(TOTAL, dtype=np.uint32)
    
    # Initialize min distances array
    min_dists = np.full(TOTAL, 100, dtype=np.uint8)
    
    # Mark seed words as used (distance 0)
    for w in code:
        min_dists[w] = 0
    
    # Calculate initial min distances using vectorized operations
    # Process in batches to manage memory
    BATCH_SIZE = 16384
    seed_list = list(code)
    for batch_start in range(0, TOTAL, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TOTAL)
        batch = all_candidates[batch_start:batch_end]
        
        # Compute distances from each candidate in batch to each codeword
        batch_min = np.full(len(batch), 100, dtype=np.uint8)
        
        for w in seed_list:
            # Vectorized XOR and popcount
            dists = np.vectorize(lambda x: (x ^ w).bit_count())(batch)
            batch_min = np.minimum(batch_min, dists)
        
        min_dists[batch_start:batch_end] = batch_min
    
    # Greedy loop: repeatedly add the candidate with maximum min-distance
    while time.time() < deadline:
        # Find the candidate with the maximum min distance that is still valid (>= MIN_DIST)
        valid_mask = min_dists >= MIN_DIST
        if not np.any(valid_mask):
            break
        
        # Find index of max min_dist among valid candidates
        valid_indices = np.where(valid_mask)[0]
        valid_dists = min_dists[valid_indices]
        best_local = np.argmax(valid_dists)
        best_idx = int(valid_indices[best_local])
        best_d = int(valid_dists[best_local])
        
        if best_d < MIN_DIST:
            break
        
        new_word = int(all_candidates[best_idx])
        code.add(new_word)
        min_dists[best_idx] = 0  # Mark as used/invalid
        
        # Update min distances for all other candidates using vectorized ops
        # Only update candidates that are still valid (min_dist >= MIN_DIST)
        new_dists = np.vectorize(lambda x: (x ^ new_word).bit_count())(all_candidates)
        min_dists = np.minimum(min_dists, new_dists)
    
    return list(code)
