"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute valid_clique and valid_indep for all masks
    valid_clique = [False] * (1 << n)
    valid_indep = [False] * (1 << n)
    valid_clique[0] = True
    valid_indep[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if valid_clique[rest] and (rest & ~adj[v]) == 0:
            valid_clique[mask] = True
        
        if valid_indep[rest] and (rest & adj[v]) == 0:
            valid_indep[mask] = True
    
    # valid_mask[m] = True if m is a clique or independent set
    valid_mask = [valid_clique[m] or valid_indep[m] for m in range(1 << n)]
    
    # DP: dp[mask] = min number of valid parts to cover exactly vertices in mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # For each mask, try all non-empty valid submasks
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if valid_mask[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
            sub = (sub - 1) & mask
    
    min_parts = dp[full_mask]
    
    # Reconstruct the partition by backtracking
    partition = []
    mask = full_mask
    while mask:
        sub = mask
        found = False
        while sub:
            if valid_mask[sub]:
                prev = mask ^ sub
                if dp[prev] == dp[mask] - 1:
                    verts = [v for v in range(n) if mask & (1 << v) and sub & (1 << v)]
                    partition.append([int(v) for v in verts])
                    mask = prev
                    found = True
                    break
            sub = (sub - 1) & mask
        if not found:
            break
    
    return partition

# EVOLVE-BLOCK-END
