"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmask for fast operations
    adj_mask = [0] * n
    for a, b in edges:
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
    
    # Precompute for every subset mask whether it is a clique or independent set
    # n=14 -> 2^14 = 16384 masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check clique: v must be adjacent to all in rest
        if is_clique[rest]:
            # v adjacent to all vertices in rest
            is_clique[mask] = (adj_mask[v] & rest) == rest
        else:
            is_clique[mask] = False
        
        # Check independent set: v must be non-adjacent to all in rest
        if is_indep[rest]:
            # v not adjacent to any vertex in rest
            is_indep[mask] = (adj_mask[v] & rest) == 0
        else:
            is_indep[mask] = False
    
    valid = [False] * (1 << n)
    for m in range(1, 1 << n):
        valid[m] = is_clique[m] or is_indep[m]
    
    full_mask = (1 << n) - 1
    
    # Precompute valid submasks for each mask to speed up DP
    # valid_submasks[mask] = list of valid submasks of mask (including mask itself)
    valid_submasks = [[] for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if valid[sub]:
                valid_submasks[mask].append(sub)
            sub = (sub - 1) & mask
    
    # DP over subsets: dp[mask] = minimum number of parts needed to cover mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n)  # parent[mask] = submask used in optimal solution
    
    for mask in range(1, 1 << n):
        for sub in valid_submasks[mask]:
            rest = mask ^ sub
            if dp[rest] + 1 < dp[mask]:
                dp[mask] = dp[rest] + 1
                parent[mask] = sub
    
    # Reconstruct solution
    result_masks = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        result_masks.append(sub)
        mask ^= sub
    
    # Convert mask list to vertex lists
    result = []
    for m in result_masks:
        verts = []
        r = m
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            verts.append(v)
            r ^= lb
        result.append(verts)
    
    # Verify result is valid
    if len(result) == 0 or any(len(p) == 0 for p in result):
        # Fallback: single vertex parts
        return [[i] for i in range(n)]
    
    return result
# EVOLVE-BLOCK-END
