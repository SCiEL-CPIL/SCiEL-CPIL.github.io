import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    N = 17
    MIN_DIST = 6
    
    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Generate generator vector g
    g = np.zeros(17, dtype=np.uint8)
    for q in Q:
        g[q] = 1
        
    # Generate basis: shifts of g
    basis = []
    for i in range(9):
        v = np.roll(g, i)
        basis.append(v)
    basis = np.array(basis)
    
    # Generate all 2^9 = 512 codewords
    codewords = []
    for mask in range(512):
        w = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        
        # Check parity
        if np.sum(w) % 2 == 0:
            val = 0
            for j in range(17):
                if w[j]:
                    val |= (1 << j)
            codewords.append(val)
            
    # Start with the 256 linear codewords
    selected = set(codewords)
    selected_list = list(selected)
    
    # Precompute popcount for all 2^17 values for fast distance checking
    # Actually, let's use a different approach: precompute distances to each selected word
    # But that's 2^17 * 256 = 33M entries, too much memory
    
    # Better approach: use bit operations for fast popcount
    # For each candidate, compute min distance to all selected
    
    # Strategy: greedy fill with vectorized distance computation
    # Process candidates in batches
    
    # Precompute a lookup table for popcount of 17-bit values
    # 2^17 = 131072 entries, each 1 byte = 128KB, fine
    popcount_table = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    # Now we can compute distance between a and b as popcount_table[a ^ b]
    
    # Greedy fill: iterate through all candidates, add if valid
    # To speed up, we can use the fact that if a candidate is too close to any selected, skip it
    
    # Let's process in batches for efficiency
    batch_size = 5000
    
    for start in range(0, 1 << N, batch_size):
        if time.time() > deadline - 3:
            break
        
        end = min(start + batch_size, 1 << N)
        candidates = np.arange(start, end, dtype=np.uint32)
        
        # Filter out already selected
        mask = np.ones(len(candidates), dtype=bool)
        for i, c in enumerate(candidates):
            if int(c) in selected:
                mask[i] = False
        
        if not np.any(mask):
            continue
        
        candidates = candidates[mask]
        if len(candidates) == 0:
            continue
        
        # For each candidate, compute min distance to all selected
        # Use vectorized approach
        # candidates: (B,)
        # selected_list: (K,)
        
        # Compute XOR for all pairs
        # This is B * K which could be large
        # B ~ 5000, K ~ 256-350
        # 5000 * 350 = 1.75M entries, fine
        
        xor_matrix = candidates[:, None] ^ np.array(selected_list, dtype=np.uint32)[None, :]
        
        # Get popcount for each entry
        # xor_matrix values are in [0, 2^17), so we can index into popcount_table
        dist_matrix = popcount_table[xor_matrix]
        
        # Min distance for each candidate
        min_dists = np.min(dist_matrix, axis=1)
        
        # Valid candidates have min_dist >= 6
        valid_mask = min_dists >= MIN_DIST
        
        # Process valid candidates one by one, re-validating against updated set
        valid_indices = np.where(valid_mask)[0]
        
        for idx in valid_indices:
            if time.time() > deadline - 1:
                break
            
            val = int(candidates[idx])
            
            # Re-validate against current selected set (which may have grown)
            is_valid = True
            for s in selected_list:
                if (val ^ s).bit_count() < MIN_DIST:
                    is_valid = False
                    break
            
            if is_valid:
                selected.add(val)
                selected_list.append(val)
    
    return list(selected)
