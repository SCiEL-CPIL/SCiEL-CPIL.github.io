import numpy as np
import time

def construct():
    deadline = time.time() + 85
    N = 18
    MIN_DIST = 6
    
    # 1. Initialize with Reed-Muller [16,5,8] extended by 2 zeros -> 32 words, dist 8
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
            
    code = np.array(base_words, dtype=np.uint32)
    
    # 2. Vectorized Greedy Addition
    rng = np.random.default_rng(42)
    
    # Target number of XOR operations per batch
    target_ops = 2_000_000
    
    while time.time() < deadline:
        # Determine batch size and chunk size
        # We want batch_size * chunk_size ~ target_ops
        # Let's fix batch_size to 10000 and adjust chunk_size
        batch_size = 10000
        chunk_size = max(1, int(target_ops / batch_size))
        
        # Generate batch of random candidates
        candidates = rng.integers(0, 2**N, size=batch_size, dtype=np.uint32)
        
        min_d = np.full(batch_size, N, dtype=np.int32)
        
        # Process code in chunks
        for i in range(0, len(code), chunk_size):
            chunk = code[i:i+chunk_size]
            if len(chunk) == 0:
                break
                
            # Compute XOR
            xors = candidates[:, None] ^ chunk[None, :]
            
            # Compute popcount
            popcounts = np.zeros(xors.shape, dtype=np.int32)
            for bit in range(18):
                popcounts += (xors >> bit) & 1
                
            # Min over chunk
            chunk_min = popcounts.min(axis=1)
            
            # Update global min
            min_d = np.minimum(min_d, chunk_min)
            
            # Optimization: If all candidates are already invalid (min_d < MIN_DIST),
            # we can break early?
            # If min_d[j] < MIN_DIST, it stays < MIN_DIST.
            # We can check if np.any(min_d >= MIN_DIST) is False.
            # But that check is O(B).
            # Let's just let it run, it's fast.
            
        # Find valid candidates
        valid_mask = min_d >= MIN_DIST
        valid_candidates = candidates[valid_mask]
        
        if len(valid_candidates) > 0:
            code = np.concatenate([code, valid_candidates])
            
    return [int(w) for w in code]
