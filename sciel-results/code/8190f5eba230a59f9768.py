"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each mask, is it a clique? is it an independent set?
    # A set is a clique if every pair has an edge
    # A set is an independent set if no pair has an edge
    
    # For clique: mask is clique iff for every v in mask, (mask without v) subset of adj[v]
    # For independent: mask is independent iff for every v in mask, (mask without v) subset of ~adj[v] (within full_mask)
    
    num_masks = 1 << n
    is_clique = [False] * num_masks
    is_independent = [False] * num_masks
    
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, num_masks):
        # Get lowest set bit
        v_bit = mask & (-mask)
        v = v_bit.bit_length() - 1
        rest = mask ^ v_bit
        
        # Check if mask is a clique: rest must be subset of adj[v], and rest must be a clique
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        
        # Check if mask is an independent set: rest must be subset of ~adj[v], and rest must be independent
        # ~adj[v] within n bits: full_mask ^ adj[v]
        non_adj = full_mask ^ adj[v]
        if is_independent[rest] and (rest & ~non_adj) == 0:
            is_independent[mask] = True
    
    # Precompute valid subsets: those that are clique OR independent (and non-empty)
    # For DP, we need for each mask, the best sub-subset to peel off
    
    # DP: dp[mask] = min number of parts to partition vertices in mask
    INF = 10**9
    dp = [INF] * num_masks
    dp[0] = 0
    # Track which sub was chosen for reconstruction
    best_sub = [0] * num_masks
    
    for mask in range(1, num_masks):
        # Try all non-empty subsets sub of mask
        # To optimize, iterate sub = mask, then sub = (sub-1) & mask
        sub = mask
        while sub:
            if is_clique[sub] or is_independent[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    best_sub[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = best_sub[mask]
        # Extract vertices from sub
        verts = []
        m = sub
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            verts.append(v)
            m ^= v_bit
        parts.append(tuple(verts))
        mask ^= sub
    
    # Convert to list of lists (required type)
    return [list(p) for p in parts]
# EVOLVE-BLOCK-END
