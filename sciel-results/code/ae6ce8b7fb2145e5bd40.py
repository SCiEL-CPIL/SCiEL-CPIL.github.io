import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute non-adjacency bitmasks (excluding self)
    # non_adj[u] has bits set for vertices not connected to u (and not u itself)
    non_adj = [0] * n
    for u in range(n):
        # All bits except u and neighbors
        non_adj[u] = full_mask ^ (1 << u) ^ adj[u]

    # Precompute is_clique and is_indep for all masks
    # is_clique[mask] is True if the induced subgraph on mask is a clique
    # is_indep[mask] is True if the induced subgraph on mask is an independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: rest must be a clique and all vertices in rest must be neighbors of u
        if is_clique[rest] and (rest & ~adj[u] == 0):
            is_clique[mask] = True
            
        # Check indep: rest must be an independent set and no vertex in rest can be a neighbor of u
        if is_indep[rest] and (rest & adj[u] == 0):
            is_indep[mask] = True
            
    # is_valid[mask] is True if mask is a clique or an independent set
    is_valid = [is_clique[m] or is_indep[m] for m in range(1 << n)]

    # DP for minimum number of parts
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    best_sub = [0] * (1 << n) # Store the submask used for reconstruction
    
    # Iterate through all masks
    for mask in range(1, 1 << n):
        # Iterate through all non-empty submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    best_sub[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct partition
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        if sub == 0:
            # Should not happen if dp is correct, but fallback to single vertex
            sub = current_mask & -current_mask
        
        # Convert sub to list of vertices
        part_vertices = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current_mask ^= sub
        
    # Ensure types are correct
    out = []
    for part in parts:
        out.append([int(v) for v in part])
        
    return out
