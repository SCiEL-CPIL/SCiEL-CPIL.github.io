import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin (90s limit, leave buffer)
    
    N = 17
    MIN_DIST = 6
    M = 1 << N  # 131072
    
    # Precompute bit matrix for all candidates (M x N)
    # This is 131072 x 17 uint8 = ~2.2 MB, fine.
    all_indices = np.arange(M, dtype=np.uint32)
    bit_positions = np.arange(N, dtype=np.uint32)
    # all_bits[m, j] = (m >> j) & 1
    # Use broadcasting: all_indices[:, None] >> bit_positions[None, :]
    all_bits = (all_indices[:, None] >> bit_positions[None, :]) & 1  # (M, N) uint8
    
    # Start with the zero word
    codewords_list = [0]
    cw_bits = np.array([all_bits[0]], dtype=np.uint8)  # (1, N)
    
    # Active mask: which candidates are not already in the code
    active = np.ones(M, dtype=bool)
    active[0] = False
    
    active_indices = np.where(active)[0]
    
    # Main greedy loop: Best-Fit strategy
    # Pick the candidate that maximizes the minimum distance to the current codewords.
    
    while time.time() < deadline:
        if len(active_indices) == 0:
            break
        
        # Find the best candidate: max(min_dist)
        # We process in chunks to manage memory and time.
        # We need to find the global max over all active indices.
        
        best_candidate = -1
        best_min_dist = -1
        
        chunk_size = 5000
        
        for i in range(0, len(active_indices), chunk_size):
            if time.time() > deadline:
                break
            chunk_idx = active_indices[i:i+chunk_size]
            chunk_bits = all_bits[chunk_idx]  # (chunk_size, N)
            # Distance to all codewords: (chunk_size, K)
            # Use broadcasting: chunk_bits[:, None, :] xor cw_bits[None, :, :]
            dists = np.bitwise_xor(chunk_bits[:, None, :], cw_bits[None, :, :]).sum(axis=2)
            min_dists = dists.min(axis=1)
            
            # Find max in this chunk
            local_max_idx = np.argmax(min_dists)
            local_max_val = min_dists[local_max_idx]
            
            if local_max_val > best_min_dist:
                best_min_dist = local_max_val
                best_candidate = chunk_idx[local_max_idx]
        
        if best_candidate == -1:
            break
            
        if best_min_dist < MIN_DIST:
            break
        
        chosen = best_candidate
        chosen_bits = all_bits[chosen]  # (N,)
        chosen_int = all_indices[chosen]
        
        # Add to codewords
        cw_bits = np.vstack([cw_bits, chosen_bits[None, :]])
        codewords_list.append(chosen_int)
        
        # Prune: remove all active candidates within distance < 6 of chosen
        # dist(chosen, c) = popcount(chosen_bits xor c_bits)
        # We need to keep only those with dist >= 6
        
        new_active_indices = []
        for i in range(0, len(active_indices), chunk_size):
            if time.time() > deadline:
                break
            chunk_idx = active_indices[i:i+chunk_size]
            chunk_bits = all_bits[chunk_idx]
            dists = np.bitwise_xor(chunk_bits, chosen_bits[None, :]).sum(axis=1)
            keep_mask = dists >= MIN_DIST
            if np.any(keep_mask):
                new_active_indices.append(chunk_idx[keep_mask])
        
        if new_active_indices:
            active_indices = np.concatenate(new_active_indices)
        else:
            active_indices = np.array([], dtype=int)
        
        # Mark chosen as inactive (though it's removed from active_indices by the logic above 
        # because dist(chosen, chosen) = 0 < 6, so it won't be in keep_mask)
        # But just to be safe, ensure it's not in active_indices if it was somehow left.
        # The pruning step removes it because dist < 6.
        
    # Build result
    result = codewords_list
    
    # Verify: all distinct, all in [0, 2^17), all pairwise dist >= 6
    assert len(set(result)) == len(result), "Duplicate codewords"
    assert all(0 <= w < 2**N for w in result), "Out of range"
    
    # Spot-check distances
    # Full check is O(K^2 * N). For K=300, 90k pairs * 17 ops ~ 1.5M ops. Fast enough.
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST} between {result[i]} and {result[j]}"
    
    return result
