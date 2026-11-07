import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 50.0
    
    n = P['n']
    edges = P['edges']
    
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    # is_clique[mask]: True if vertices in mask form a clique
    # is_independent[mask]: True if vertices in mask form an independent set
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest
        # adj_masks[v] contains neighbors of v
        # (adj_masks[v] & rest) == rest means v connected to all in rest
        if is_clique[rest] and (adj_masks[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: v must NOT be connected to any in rest
        # (adj_masks[v] & rest) == 0 means v not connected to any in rest
        if is_independent[rest] and (adj_masks[v] & rest) == 0:
            is_independent[mask] = True
            
    # DP
    dp = [float('inf')] * size
    parent = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        if time.time() > deadline:
            break
            
        # Iterate submasks of mask
        # We only care about non-empty submasks that are valid parts
        s = mask
        while s:
            if is_clique[s] or is_independent[s]:
                prev = mask ^ s
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = s
            s = (s - 1) & mask
            
    # Fallback if timeout or failure
    if time.time() > deadline or dp[full_mask] == float('inf'):
        # Greedy fallback
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        s = parent[mask]
        if s == -1:
            break
        part = []
        for i in range(n):
            if s & (1 << i):
                part.append(i)
        parts.append(part)
        mask ^= s
        
    return parts
