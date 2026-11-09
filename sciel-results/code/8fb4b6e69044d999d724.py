import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute valid subsets (cliques or independent sets)
    # Group them by their lowest set bit (pivot) to speed up DP
    valid_subs_by_pivot = [[] for _ in range(n)]
    full_mask = (1 << n) - 1
    
    for mask in range(1, full_mask + 1):
        is_clique = True
        is_indep = True
        
        # Check if clique
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if (mask & ~adj_mask[v] & ~(1 << v)) != 0:
                is_clique = False
                break
            m -= lsb
        
        if is_clique:
            lsb = mask & -mask
            pivot = lsb.bit_length() - 1
            valid_subs_by_pivot[pivot].append(mask)
            continue
        
        if not is_clique:
            # Check if independent
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if (mask & adj_mask[v]) != 0:
                    is_indep = False
                    break
                m -= lsb
            
            if is_indep:
                lsb = mask & -mask
                pivot = lsb.bit_length() - 1
                valid_subs_by_pivot[pivot].append(mask)
                
    # DP over subsets
    # dp[mask] = min number of parts to cover vertices in 'mask'
    size = 1 << n
    dp = [float('inf')] * size
    choice = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Pick a pivot vertex (lowest set bit) to break symmetry
        lsb = mask & -mask
        pivot = lsb.bit_length() - 1
        
        best_val = float('inf')
        best_sub = -1
        
        # Only iterate through valid subsets that have this pivot
        for sub in valid_subs_by_pivot[pivot]:
            # Check if sub is a subset of mask
            if (sub & mask) == sub:
                new_mask = mask ^ sub
                val = 1 + dp[new_mask]
                if val < best_val:
                    best_val = val
                    best_sub = sub
        
        dp[mask] = best_val
        choice[mask] = best_sub
        
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == -1:
            break
        part = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(v)
            m -= lsb
        partition.append(part)
        mask ^= sub
        
    return partition

# EVOLVE-BLOCK-END
