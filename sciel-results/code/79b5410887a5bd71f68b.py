def construct():
    import time
    import numpy as np
    import random

    start_time = time.time()
    deadline = start_time + 85
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Helper: popcount for numpy array of ints
    # We can use np.bitwise_xor and then sum bits?
    # Or use a lookup table for bytes?
    # For 17-bit ints, we can split into 8+8+1 or just use a lookup table for 0..255.
    # Better: use np.unpackbits? No, that's for bytes.
    # Let's use a lookup table for 0..255.
    popcount_8 = np.array([i.bit_count() for i in range(256)], dtype=np.int8)
    
    def popcount_17(arr):
        # arr is numpy array of ints
        # Split into low 8 bits, mid 8 bits, high 1 bit
        # Actually 17 bits fits in int32.
        # We can do:
        # p = popcount_8[arr & 0xFF] + popcount_8[(arr >> 8) & 0xFF] + popcount_8[(arr >> 16) & 0xFF]
        return popcount_8[arr & 0xFF] + popcount_8[(arr >> 8) & 0xFF] + popcount_8[(arr >> 16) & 0xFF]

    # Phase 1: Find 256-word linear code
    # Greedy random basis selection
    # We need to generate all vectors? No, just scan.
    # To be fast, we can just try to build a basis.
    
    # Let's try to find a basis b1...b8
    # We will maintain the code as a list of ints.
    # But vectorized check is better.
    
    # Let's generate all vectors as a numpy array? 131k ints.
    all_vectors = np.arange(SIZE, dtype=np.int32)
    
    # Shuffle indices for randomization
    # We want to find a linear code.
    # Greedy:
    # 1. Pick a random vector of weight >= 6.
    # 2. Check if it's orthogonal to current basis? No, just check distance.
    
    # Actually, the current program's greedy linear code is fine if optimized.
    # But let's use a simpler heuristic:
    # Try to find 8 linearly independent vectors such that the code they generate has min distance 6.
    
    # We can use a random search for the basis.
    # But there's a risk of failing.
    # Let's use the deterministic high-weight sort from the current program, but vectorize the check.
    
    # Sort indices by weight descending
    weights_all = popcount_17(all_vectors)
    indices = np.argsort(-weights_all)
    
    basis = []
    code = [0]
    
    # We need to check if g can be added.
    # d(g, c) >= 6 for all c in code.
    
    for idx in indices:
        if time.time() > deadline - 10: break
        if len(basis) >= 8: break
        
        g = int(all_vectors[idx])
        
        # Check distance
        # code is a list of ints.
        # Convert to numpy array?
        # If code is small, python loop is okay.
        # Max code size 256.
        valid = True
        for c in code:
            if popcount_17(np.array([g ^ c]))[0] < MIN_D:
                valid = False
                break
        
        if valid:
            basis.append(g)
            new_words = [c ^ g for c in code]
            code.extend(new_words)
            
    if len(code) < 256:
        # Fallback: return what we have
        return code

    # Phase 2: Extend
    C_np = np.array(code, dtype=np.int32)
    
    # Find all vectors at distance >= 6 from C_np
    # We process in chunks to save memory
    valid_vectors = []
    chunk_size = 1000
    for i in range(0, SIZE, chunk_size):
        if time.time() > deadline - 5: break
        chunk = all_vectors[i:i+chunk_size]
        # Compute distances
        # dists[j, k] = d(chunk[j], C_np[k])
        # We can do this by broadcasting
        # chunk[:, None] ^ C_np[None, :]
        # This creates a (1000, 256) array.
        xors = chunk[:, None] ^ C_np[None, :]
        pcounts = popcount_17(xors)
        min_dists = pcounts.min(axis=1)
        
        # Find indices where min_dist >= 6
        mask = min_dists >= MIN_D
        valid_in_chunk = chunk[mask]
        valid_vectors.extend(valid_in_chunk.tolist())
        
    # Now we have a list of valid vectors.
    # We want to select a maximal subset with pairwise distance >= 6.
    # Greedy selection.
    
    selected = []
    # To improve results, we can try multiple random orders.
    # But we have limited time.
    # Let's try a few random shuffles of valid_vectors.
    
    # valid_vectors size?
    # Let's estimate.
    # If it's large, we need to be efficient.
    
    # We'll do up to 10 random restarts of the greedy selection.
    best_ext = []
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 2: break
        
        random.shuffle(valid_vectors)
        current_selected = []
        
        for v in valid_vectors:
            if time.time() > deadline - 2: break
            
            # Check if v is compatible with current_selected
            # current_selected is small (maybe < 50)
            valid = True
            for c in current_selected:
                if popcount_17(np.array([v ^ c]))[0] < MIN_D:
                    valid = False
                    break
            
            if valid:
                current_selected.append(v)
        
        if len(current_selected) > len(best_ext):
            best_ext = current_selected
            
    # Combine
    final_code = code + best_ext
    return final_code
