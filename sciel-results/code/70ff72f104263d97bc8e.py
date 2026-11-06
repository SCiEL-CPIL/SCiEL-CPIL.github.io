import numpy as np
import time

def construct():
    N = 18
    MIN_D = 6
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    codebook = []
    total_candidates = 2 ** N
    chunk_size = 5000
    start_idx = 0
    
    while start_idx < total_candidates:
        if time.time() > deadline:
            break
        
        end_idx = min(start_idx + chunk_size, total_candidates)
        candidates_chunk = np.arange(start_idx, end_idx, dtype=np.uint32)
        
        if not codebook:
            # If codebook is empty, the first candidate is always valid
            codebook.append(int(candidates_chunk[0]))
            start_idx += 1
            continue
        
        # Convert current codebook to numpy array for vectorized operations
        cb_arr = np.array(codebook, dtype=np.uint32)
        K = len(cb_arr)
        
        # Calculate XOR distances
        # candidates_chunk shape: (M,)
        # cb_arr shape: (K,)
        # We want (M, K)
        xors = candidates_chunk[:, None] ^ cb_arr[None, :]
        
        # Calculate population count (Hamming weight)
        # View as uint8 to use unpackbits
        # xors is uint32, so 4 bytes per element
        bytes_view = xors.view(np.uint8).reshape(-1, 4)
        bits = np.unpackbits(bytes_view, axis=1)
        
        # Sum only the first 18 bits
        dists = bits[:, :18].sum(axis=1).reshape(-1, K)
        
        # Find candidates with minimum distance >= MIN_D
        min_dists = dists.min(axis=1)
        valid_mask = min_dists >= MIN_D
        
        valid_candidates = candidates_chunk[valid_mask]
        
        # Add valid candidates to codebook one by one to ensure pairwise distance
        # among the newly added ones
        newly_added = []
        for c in valid_candidates:
            if time.time() > deadline:
                break
            c_int = int(c)
            # Check against newly added in this batch
            is_valid = True
            for na in newly_added:
                if (c_int ^ na).bit_count() < MIN_D:
                    is_valid = False
                    break
            if is_valid:
                codebook.append(c_int)
                newly_added.append(c_int)
        
        start_idx = end_idx
        
    return codebook
