import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
    
    is_valid = [False] * size
    for m in range(1, size):
        is_valid[m] = is_clique[m] or is_independent[m]
    
    # Exact DP: dp[mask] = minimum number of valid parts to partition vertices in mask
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    parent = [0] * size  # parent[mask] = the valid part chosen to reach mask
    
    for mask in range(1, size):
        # Try all valid submasks S of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    if dp[full_mask] == INF:
        # Fallback: shouldn't happen since single vertices are always valid
        return [[i] for i in range(n)]
    
    # Reconstruct partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == 0:
            break
        verts = []
        m = sub
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        parts.append(verts)
        mask ^= sub
    
    # Ensure all vertices are present exactly once
    all_verts = []
    for part in parts:
        all_verts.extend(part)
    
    # Verify and return as list of lists of Python ints
    result = []
    for part in parts:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
