"""Rewritten program using dynamic programming over subsets for optimal cochromatic partition."""

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def construct():
    _t0 = time.time()
    _deadline = 75.0  # seconds, safety margin before 90s kill
    
    n = P['n']
    
    # Precompute adjacency bitmask
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute which subsets are cliques and which are independent sets
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    # Empty set and singletons are both clique and independent
    is_clique[0] = True
    is_indep[0] = True
    for i in range(n):
        mask = 1 << i
        is_clique[mask] = True
        is_indep[mask] = True
        
    # Fill DP for clique/independent property
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue # Single bit already handled
        
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
        
        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True
    
    # Precompute valid: subset is a valid part if clique or independent
    valid = [is_clique[m] or is_indep[m] for m in range(1 << n)]
    
    # DP for minimum number of parts
    INF = float('inf')
    dp = [INF] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        if time.time() - _t0 > _deadline:
            # Fallback: greedy construction
            break
        sub = mask
        best_val = dp[mask]
        best_sub = 0
        while sub:
            if valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        dp[mask] = best_val
        choice[mask] = best_sub
    
    # Check if DP completed
    full_mask = (1 << n) - 1
    if dp[full_mask] == INF:
        # Fallback: each vertex as its own part
        return [[i] for i in range(n)]
    
    # Reconstruct the partition
    parts = []
    current_mask = full_mask
    while current_mask:
        part_mask = choice[current_mask]
        part_vertices = [i for i in range(n) if (part_mask >> i) & 1]
        parts.append(part_vertices)
        current_mask ^= part_mask
        
    # Ensure all elements are Python int
    result = [list(p) for p in parts]
    for part in result:
        for i in range(len(part)):
            part[i] = int(part[i])
            
    return result
