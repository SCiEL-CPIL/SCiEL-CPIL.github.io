def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    for mask in range(1 << n):
        verts = [i for i in range(n) if mask & (1 << i)]
        if len(verts) <= 1:
            is_clique[mask] = True
            is_indep[mask] = True
        else:
            clique = True
            indep = True
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if adj[u] & (1 << v):
                        indep = False
                    else:
                        clique = False
            is_clique[mask] = clique
            is_indep[mask] = indep
    
    valid = [is_clique[m] or is_indep[m] for m in range(1 << n)]
    
    full_mask = (1 << n) - 1
    best_count = n + 1
    
    # Precompute for each v, the list of valid masks containing v, sorted by size descending
    valid_masks_containing = [[] for _ in range(n)]
    for mask in range(1 << n):
        if not valid[mask]:
            continue
        for v in range(n):
            if mask & (1 << v):
                valid_masks_containing[v].append(mask)
    
    for v in range(n):
        valid_masks_containing[v].sort(key=lambda m: m.bit_count(), reverse=True)
    
    def backtrack(remaining, count):
        nonlocal best_count
        if count >= best_count:
            return False
        if remaining == 0:
            best_count = count
            return True
        
        v = (remaining & -remaining).bit_length() - 1
        rem_wo_v = remaining ^ (1 << v)
        
        for mask in valid_masks_containing[v]:
            if mask & ~remaining:
                continue
            if not (mask & (1 << v)):
                continue
            if backtrack(remaining ^ mask, count + 1):
                return True
        
        return False
    
    backtrack(full_mask, 0)
    
    # Reconstruct
    parts = []
    def reconstruct(remaining, count):
        if count == 0:
            return remaining == 0
        if remaining == 0:
            return False
        
        v = (remaining & -remaining).bit_length() - 1
        
        for mask in valid_masks_containing[v]:
            if mask & ~remaining:
                continue
            if not (mask & (1 << v)):
                continue
            part = [i for i in range(n) if mask & (1 << i)]
            parts.append(part)
            if reconstruct(remaining ^ mask, count - 1):
                return True
            parts.pop()
        
        return False
    
    reconstruct(full_mask, best_count)
    
    return parts
