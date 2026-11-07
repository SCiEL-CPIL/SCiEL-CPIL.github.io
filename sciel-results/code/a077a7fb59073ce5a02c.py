import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    # Precompute candidates for each vertex
    # cands[v] contains masks that are valid, contain v, sorted by size desc
    cands = [[] for _ in range(n)]
    for m in range(1, 1 << n):
        if is_valid[m]:
            # Find vertices in m
            temp = m
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                cands[v].append(m)
                temp ^= lsb
                
    # Sort by size descending
    for v in range(n):
        cands[v].sort(key=lambda x: x.bit_count(), reverse=True)
        
    # DFS
    best_depth = n + 1
    best_path = None
    
    # Optional: Greedy upper bound to prune early
    # Greedy: pick largest valid set containing lowest bit
    def greedy_upper_bound():
        rem = full_mask
        parts = []
        while rem:
            lsb = rem & -rem
            v = lsb.bit_length() - 1
            found = False
            for sub in cands[v]:
                if (sub & rem) == sub:
                    parts.append(sub)
                    rem ^= sub
                    found = True
                    break
            if not found:
                # Should not happen if we have single vertices
                parts.append(lsb)
                rem ^= lsb
        return len(parts), parts

    g_depth, g_path = greedy_upper_bound()
    best_depth = g_depth
    best_path = g_path
    
    def dfs(rem, depth, path):
        nonlocal best_depth, best_path
        if rem == 0:
            if depth < best_depth:
                best_depth = depth
                best_path = path[:]
            return
            
        if depth >= best_depth:
            return
            
        # Lower bound: at least 1 if rem != 0
        if depth + 1 >= best_depth:
            return
            
        lsb = rem & -rem
        v = lsb.bit_length() - 1
        
        for sub in cands[v]:
            if (sub & rem) == sub:
                path.append(sub)
                dfs(rem ^ sub, depth + 1, path)
                path.pop()
                if best_depth <= depth: # Prune if we found a better solution
                    return

    dfs(full_mask, 0, [])
    
    # Convert masks to lists of ints
    def mask_to_list(mask):
        verts = []
        while mask:
            b = mask & -mask
            verts.append(b.bit_length() - 1)
            mask ^= b
        return verts

    if best_path is None:
        # Fallback (should not happen)
        return [[i] for i in range(n)]
        
    result = [mask_to_list(m) for m in best_path]
    return result
