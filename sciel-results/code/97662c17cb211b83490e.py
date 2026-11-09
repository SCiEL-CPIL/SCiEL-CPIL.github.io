import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit
    
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute non-adjacency bitmasks (excluding self)
    non_adj = [0] * n
    for u in range(n):
        non_adj[u] = full_mask ^ (1 << u) ^ adj[u]

    # Precompute is_clique and is_indep for all masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[u] == 0):
            is_clique[mask] = True
            
        if is_indep[rest] and (rest & adj[u] == 0):
            is_indep[mask] = True
            
    # is_valid[mask] is True if mask is a clique or an independent set
    is_valid = [False] * (1 << n)
    for m in range(1 << n):
        is_valid[m] = is_clique[m] or is_indep[m]

    # DP for minimum number of parts
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    best_sub = [0] * (1 << n)
    
    # Iterate through all masks
    for mask in range(1, 1 << n):
        # Check deadline periodically
        if mask % 1024 == 0 and time.time() > deadline:
            # Fall back to greedy
            return _greedy_partition(n, adj, full_mask)
            
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    best_sub[mask] = sub
            sub = (sub - 1) & mask
            
    # Check final deadline
    if time.time() > deadline:
        return _greedy_partition(n, adj, full_mask)
        
    # Reconstruct partition
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        if sub == 0:
            sub = current_mask & -current_mask
        
        part_vertices = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current_mask ^= sub
        
    out = []
    for part in parts:
        out.append([int(v) for v in part])
        
    return out

def _greedy_partition(n, adj, full_mask):
    """Fallback: greedy construction of a valid cochromatic partition."""
    remaining = set(range(n))
    parts = []
    
    while remaining:
        # Try to build a large clique or independent set from remaining
        best_part = None
        best_size = 0
        
        # Try each vertex as seed
        for seed in list(remaining):
            # Try clique
            clique = {seed}
            for v in list(remaining - clique):
                if all((adj[v] >> u) & 1 for u in clique):
                    clique.add(v)
            if len(clique) > best_size:
                best_size = len(clique)
                best_part = clique
            
            # Try independent set
            indep = {seed}
            for v in list(remaining - indep):
                if all(not ((adj[v] >> u) & 1) for u in indep):
                    indep.add(v)
            if len(indep) > best_size:
                best_size = len(indep)
                best_part = indep
        
        if best_part is None:
            best_part = {min(remaining)}
        
        parts.append([int(v) for v in best_part])
        remaining -= best_part
    
    return parts
