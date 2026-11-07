import time

def construct():
    # Data from the seed program
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all subsets
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    # Helper to convert mask to list of vertices
    def mask_to_list(mask):
        verts = []
        while mask:
            b = mask & -mask
            verts.append(b.bit_length() - 1)
            mask ^= b
        return verts

    # Bottom-up DP for minimum number of parts
    # dp[mask] = min parts to cover 'mask'
    # choice[mask] = the submask chosen for 'mask' in the optimal solution
    dp = [n + 1] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0
    
    # We iterate masks from 1 to full_mask.
    # To ensure that when we compute dp[mask], dp[rest] is already computed,
    # we don't strictly need topological order if we iterate all submasks?
    # dp[mask] = 1 + min(dp[mask ^ sub]) for sub in valid_submasks(mask) containing lsb(mask).
    # Since sub is a submask of mask, mask ^ sub is a proper submask of mask.
    # If we iterate masks in increasing order of integer value, does mask ^ sub < mask?
    # Yes, mask ^ sub clears some bits of mask, so its integer value is strictly smaller.
    # So simple iteration from 1 to (1<<n)-1 works.
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        best_val = n + 1
        best_sub = 0
        
        # Iterate submasks of mask containing lsb
        # We can iterate all submasks and check if they contain lsb, 
        # or iterate submasks of (mask ^ lsb) and add lsb.
        # The latter is more efficient as it halves the search space.
        rest_mask = mask ^ lsb
        sub_rest = rest_mask
        while True:
            sub = sub_rest | lsb
            if is_valid[sub]:
                rem = mask ^ sub
                val = 1 + dp[rem]
                if val < best_val:
                    best_val = val
                    best_sub = sub
            
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest_mask
        
        dp[mask] = best_val
        choice[mask] = best_sub
        
    optimal_k = dp[full_mask]
    
    # Reconstruct
    current_parts = []
    rem = full_mask
    while rem:
        sub = choice[rem]
        if sub == 0:
            # Should not happen if dp[full_mask] is valid
            # Fallback: single vertex
            sub = rem & -rem
        current_parts.append(mask_to_list(sub))
        rem ^= sub
        
    # Ensure types are correct
    # The lists contain Python ints from bit_length
    return current_parts
