import numpy as np
import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    
    # 1. Initialize with a greedy linear code
    # Start with the zero word
    codes = [0]
    
    # Helper to convert int to bit array
    def int_to_bits(x, n):
        return np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)
    
    # Greedy phase: Add random words that are far enough
    # We want to reach at least 256
    max_greedy_steps = 10000
    for _ in range(max_greedy_steps):
        if len(codes) >= 256:
            break
        # Try to add a new word
        # Generate a batch of candidates
        batch_size = 100
        candidates = [random.getrandbits(N) for _ in range(batch_size)]
        
        # Convert to numpy for fast distance
        cand_arr = np.array([int_to_bits(c, N) for c in candidates])
        code_arr = np.array([int_to_bits(c, N) for c in codes])
        
        # Calculate distances
        # (K, 1, N) ^ (1, M, N)
        dists = np.sum(cand_arr[:, None, :] ^ code_arr[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        
        valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        if len(valid_indices) > 0:
            # Add the first valid candidate
            new_word = candidates[valid_indices[0]]
            codes.append(new_word)

    # Convert to numpy array for search phase
    codes_arr = np.array([int_to_bits(c, N) for c in codes], dtype=np.uint8)
    current_size = len(codes_arr)
    
    # 2. Local Search Phase
    start_time = time.time()
    timeout = 55
    
    while time.time() - start_time < timeout:
        # Try to add
        batch_size = 500
        candidates = np.random.randint(0, 2, size=(batch_size, N), dtype=np.uint8)
        
        # Calculate distances
        dists = np.sum(candidates[:, None, :] ^ codes_arr[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        
        valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        if len(valid_indices) > 0:
            # Add all valid ones that are mutually far?
            # For simplicity, add one, then re-evaluate
            # Or add all if they are far apart from each other
            # Let's just add the first one to be safe and fast
            new_word = candidates[valid_indices[0]]
            codes_arr = np.vstack([codes_arr, new_word])
            current_size += 1
        else:
            # Try swap
            # Pick a random index to remove
            idx_to_remove = np.random.randint(0, current_size)
            
            # Try to find a replacement
            # Generate more candidates
            candidates = np.random.randint(0, 2, size=(batch_size, N), dtype=np.uint8)
            
            # Check against codes WITHOUT idx_to_remove
            mask = np.ones(current_size, dtype=bool)
            mask[idx_to_remove] = False
            remaining_codes = codes_arr[mask]
            
            dists = np.sum(candidates[:, None, :] ^ remaining_codes[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            valid_indices = np.where(min_dists >= MIN_DIST)[0]
            
            if len(valid_indices) > 0:
                new_word = candidates[valid_indices[0]]
                # Replace
                codes_arr[idx_to_remove] = new_word
                # Size remains same

    # Convert back to list of integers
    result = []
    for i in range(len(codes_arr)):
        word = 0
        for j in range(N):
            if codes_arr[i][j]:
                word |= (1 << j)
        result.append(word)
        
    return result
