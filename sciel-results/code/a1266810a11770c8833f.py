import numpy as np
import random
import time

def construct():
    # 1. Generate Seed (same as current)
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
    
    # Convert seed to bits for NumPy
    seed_bits = np.array([[ (w >> i) & 1 for i in range(17)] for w in seed_words], dtype=np.uint8)
    
    # 2. Generate Pool (all 17-bit words)
    # We can filter out words that are too close to the seed initially?
    # Or just use all. 131072 is small.
    all_words = np.arange(1 << 17, dtype=np.int32)
    # Convert to bits
    # Efficient conversion:
    # Create a lookup table for 8-bit chunks? Or just use bit shifts.
    # 131k * 17 operations is fine.
    pool_bits = np.zeros((1 << 17, 17), dtype=np.uint8)
    for i in range(17):
        pool_bits[:, i] = (all_words >> i) & 1
        
    # Remove seed words from pool to avoid duplicates
    seed_set = set(seed_words)
    pool_mask = np.array([w not in seed_set for w in all_words])
    pool_bits = pool_bits[pool_mask]
    pool_ints = all_words[pool_mask]
    
    best_code = seed_words[:]
    best_len = len(seed_words)
    
    # 3. Greedy Search with NumPy
    # We do multiple restarts with shuffled pool
    num_restarts = 10 # Adjust based on time
    for r in range(num_restarts):
        random.shuffle(pool_ints) # Shuffle ints, but we need to shuffle bits too?
        # Easier: create a random permutation of indices
        indices = np.random.permutation(len(pool_ints))
        
        current_code_bits = seed_bits.copy()
        current_code_ints = set(seed_words)
        
        # Greedy loop
        for idx in indices:
            w_bits = pool_bits[idx]
            w_int = pool_ints[idx]
            
            if w_int in current_code_ints:
                continue
                
            # Check distance to all current codewords
            # XOR and sum
            # current_code_bits is (K, 17). w_bits is (17,).
            # Broadcast: (K, 17) ^ (1, 17) -> (K, 17)
            diffs = current_code_bits ^ w_bits[None, :]
            dists = np.sum(diffs, axis=1)
            
            if np.min(dists) >= 6:
                current_code_ints.add(w_int)
                current_code_bits = np.vstack([current_code_bits, w_bits[None, :]])
                
        if len(current_code_ints) > best_len:
            best_len = len(current_code_ints)
            best_code = list(current_code_ints)
            if best_len > 300:
                break
                
    return best_code
