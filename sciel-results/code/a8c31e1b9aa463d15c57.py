# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Precompute adjacency as bitmask
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if (i, j) in edge_set:
                adj[i] |= (1 << j)
                
    # Precompute is_clique and is_indep for all masks
    # is_clique[mask]: True if mask induces a clique
    # is_indep[mask]: True if mask induces an independent set
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        # Get least significant bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: v must NOT be connected to any in rest
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    # Collect all valid submasks (cliques or independent sets)
    valid_submasks = []
    for mask in range(1, size):
        if is_clique[mask] or is_indep[mask]:
            valid_submasks.append(mask)
    
    # DP: dp[mask] = min parts to cover mask
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    choice = [0] * size # store the submask chosen
    
    for mask in range(1, size):
        best = INF
        best_sub = 0
        # Iterate over all valid submasks that are subsets of mask
        for sub in valid_submasks:
            if (sub & mask) == sub:
                prev = dp[mask ^ sub]
                if prev + 1 < best:
                    best = prev + 1
                    best_sub = sub
        
        dp[mask] = best
        choice[mask] = best_sub
        
    # Reconstruct
    parts = []
    mask = (1 << n) - 1
    while mask:
        sub = choice[mask]
        if sub == 0:
            # This shouldn't happen if dp is correct, but safety check
            break
        verts = []
        m = sub
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        parts.append(verts)
        mask ^= sub
        
    return parts
# EVOLVE-BLOCK-END
