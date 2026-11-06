import numpy as np
import time

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55.0  # Safety margin for 60s limit

    # Helper to convert int to bit array (vectorized)
    def ints_to_bits(arr):
        # arr: 1D array of uint32
        # Returns 2D array of uint8 (len(arr), N)
        return (arr[:, None] >> np.arange(N, dtype=np.uint32)) & 1

    # 1. Initialize with Reed-Muller RM(1,4) extended to 18 bits
    # RM(1,4) has 32 codewords of length 16, min distance 8.
    # We pad with 2 zeros to make length 18. Min distance remains 8.
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    code = list(base_words)
    
    # Prepare candidates
    # Use weight ordering to find dense packing early
    all_candidates = list(range(1 << N))
    all_candidates.sort(key=lambda x: x.bit_count())
    
    # Remove candidates already in code to save time (optional, but good)
    # Actually, just checking distance is fine.
    
    # Maintain current code as list and numpy array of bits
    code_np = np.array(ints_to_bits(np.array(code, dtype=np.uint32)), dtype=np.uint8)
    current_size = len(code)
    best_code = list(code)
    best_size = current_size
    
    # Process candidates in batches
    batch_size = 1024
    i = 0
    total_candidates = len(all_candidates)
    
    while i < total_candidates and time.time() < deadline - 2.0:
        batch = all_candidates[i:i+batch_size]
        i += batch_size
        
        # Convert batch to bits
        batch_arr = np.array(batch, dtype=np.uint32)
        batch_bits = ints_to_bits(batch_arr)
        
        # Compute distances to all current codewords
        # batch_bits: (B, N)
        # code_np: (K, N)
        # XOR: (B, 1, N) ^ (1, K, N) -> (B, K, N)
        # Sum: (B, K)
        
        # To reduce memory, if K is large, we might need to split code_np
        # But K is likely < 512. B=1024. 1024*512*18*1 bytes ~ 9MB. Safe.
        
        xor = batch_bits[:, None, :] ^ code_np[None, :, :]
        dists = xor.sum(axis=2)
        
        # Find candidates with min distance >= MIN_DIST
        valid_mask = (dists >= MIN_DIST).all(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            new_words = batch[valid_indices]
            # Add to code
            code.extend(new_words)
            
            # Update code_np
            new_bits = ints_to_bits(np.array(new_words, dtype=np.uint32))
            code_np = np.vstack([code_np, new_bits])
            
            current_size = len(code)
            if current_size > best_size:
                best_size = current_size
                best_code = list(code)
                
    return best_code
