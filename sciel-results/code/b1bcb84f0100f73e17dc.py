import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    full_mask = (1 << n) - 1
    
    # 1. Build adjacency masks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # 2. Precompute valid subsets (Clique or Independent Set)
    # We only need to know if a mask is valid.
    # valid[mask] is True if mask is a clique or independent set.
    
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_independent[rest] and (adj[v] & rest) == 0:
            is_independent[mask] = True

    valid = [False] * (1 << n)
    for mask in range(1 << n):
        valid[mask] = is_clique[mask] or is_independent[mask]
        
    # 3. DP
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        sub = mask
        while sub:
            if sub & lsb:
                if valid[sub]:
                    rest = mask ^ sub
                    if dp[rest] + 1 < dp[mask]:
                        dp[mask] = dp[rest] + 1
                        parent[mask] = sub
            sub = (sub - 1) & mask
            
            # Safety check (should never trigger for n=14)
            if time.time() - start_time > 80:
                 # Fallback to greedy if somehow slow? 
                 # But it shouldn't be.
                 pass

    # 4. Reconstruction
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        verts = []
        for i in range(n):
            if sub & (1 << i):
                verts.append(i)
        parts.append(verts)
        mask ^= sub
        
    return parts
