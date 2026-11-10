import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # non_adj[i] is a bitmask of non-neighbors of i (excluding i itself)
    # Full mask for n bits
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        # All bits set except i and adj[i]
        # ~adj[i] & full_mask & ~(1<<i)
        non_adj[i] = full_mask & ~(adj[i] | (1 << i))
        
    # Precompute is_clique and is_indep for all masks
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_valid = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    is_valid[0] = True
    
    for i in range(n):
        mask = 1 << i
        is_clique[mask] = True
        is_indep[mask] = True
        is_valid[mask] = True
        
    for mask in range(1, size):
        if is_valid[mask]:
            continue # Already processed (single bits)
            
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        # rest must be clique, and v must be connected to all in rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set
        # rest must be independent, and v must be non-connected to all in rest
        if is_indep[rest] and (non_adj[v] & rest) == rest:
            is_indep[mask] = True
            
        is_valid[mask] = is_clique[mask] or is_indep[mask]
        
    # DP for minimum number of parts
    dp = [1000] * size
    dp[0] = 0
    
    for S in range(1, size):
        best = 1000
        T = S
        while T:
            if is_valid[T]:
                val = 1 + dp[S ^ T]
                if val < best:
                    best = val
            T = (T - 1) & S
        dp[S] = best
        
        # Time check
        if time.time() > deadline:
            # If we run out of time, we might have incomplete DP.
            # But 3^14 is fast. This is just a guard.
            pass

    # Backtrack to find the partition
    S = full_mask
    parts = []
    while S:
        T = S
        found = False
        while T:
            if is_valid[T] and dp[S] == 1 + dp[S ^ T]:
                # Extract vertices from T
                part_vertices = []
                tt = T
                while tt:
                    lsb = tt & -tt
                    v = lsb.bit_length() - 1
                    part_vertices.append(v)
                    tt ^= lsb
                parts.append(part_vertices)
                S ^= T
                found = True
                break
            T = (T - 1) & S
        if not found:
            # This shouldn't happen if DP is correct
            break
            
    return parts

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
