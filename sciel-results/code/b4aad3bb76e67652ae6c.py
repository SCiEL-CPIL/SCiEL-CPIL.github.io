<<<<<<< SEARCH
def _find_clique(A, Pmask, k, node_budget, deadline):
    n = len(A)
    if n < k:
        return None
    # Build adjacency and compute degrees
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    # Compute degree of each vertex
    deg = [adj[i].bit_count() for i in range(n)]
    
    # Only keep vertices with sufficient degree
    min_deg_needed = k - 1
    active = [i for i in range(n) if deg[i] >= min_deg_needed]
    if len(active) < k:
        return None
    
    # Remap to 0..m-1 where m = len(active)
    m = len(active)
    old_to_new = {old: new for new, old in enumerate(active)}
    
    # Build new adjacency in compact form
    new_adj = [0]*m
    for i in range(m):
        for j in range(i+1, m):
            oi, oj = active[i], active[j]
            if adj[oi] & (1 << oj):
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
    
    # Precompute higher masks
    higher = [0]*m
    for v in range(m-2, -1, -1):
        higher[v] = higher[v+1] | (1 << (v+1))
    
    nodes = [0]
    aborted = [False]
    import time as _t
    
    def dfs(cand_mask, need, path):
        if need == 0:
            return list(path)
        if cand_mask.bit_count() < need:
            return None
        nodes[0] += 1
        if nodes[0] > node_budget or _t.time() > deadline:
            aborted[0] = True
            return None
        
        # Get candidates sorted by degree (within current candidate set)
        msk = cand_mask
        cands = []
        while msk:
            lsb = msk & (-msk)
            v = lsb.bit_length() - 1
            nc_count = (cand_mask & new_adj[v] & higher[v]).bit_count()
            if nc_count + 1 >= need:
                cands.append((nc_count, v))
            msk &= msk - 1
        
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            nc = cand_mask & new_adj[v] & higher[v]
            res = dfs(nc, need-1, path + [v])
            if res is not None:
                return res
        
        return None
    
    res = dfs((1<<m)-1, k, [])
    if res is None:
        return None
    vals = [A[active[v]] for v in res]
    # verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals
=======
def _find_clique(A, Pmask, k, node_budget, deadline):
    n = len(A)
    if n < k:
        return None
    # Build adjacency and compute degrees
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    # Compute degree of each vertex
    deg = [adj[i].bit_count() for i in range(n)]
    
    # Only keep vertices with sufficient degree
    min_deg_needed = k - 1
    active = [i for i in range(n) if deg[i] >= min_deg_needed]
    if len(active) < k:
        return None
    
    # Sort active by degree descending for better search ordering
    active.sort(key=lambda i: -deg[i])
    
    # Remap to 0..m-1 where m = len(active)
    m = len(active)
    
    # Build new adjacency in compact form
    new_adj = [0]*m
    for i in range(m):
        for j in range(i+1, m):
            oi, oj = active[i], active[j]
            if adj[oi] & (1 << oj):
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
    
    # Precompute higher masks
    higher = [0]*m
    for v in range(m-2, -1, -1):
        higher[v] = higher[v+1] | (1 << (v+1))
    
    nodes = [0]
    aborted = [False]
    
    def dfs(cand_mask, need, path):
        if need == 0:
            return list(path)
        if cand_mask.bit_count() < need:
            return None
        nodes[0] += 1
        if nodes[0] > node_budget or time.time() > deadline:
            aborted[0] = True
            return None
        
        # Get candidates sorted by degree (within current candidate set)
        msk = cand_mask
        cands = []
        while msk:
            lsb = msk & (-msk)
            v = lsb.bit_length() - 1
            nc_count = (cand_mask & new_adj[v] & higher[v]).bit_count()
            if nc_count + 1 >= need:
                cands.append((nc_count, v))
            msk &= msk - 1
        
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            nc = cand_mask & new_adj[v] & higher[v]
            res = dfs(nc, need-1, path + [v])
            if res is not None:
                return res
        
        return None
    
    res = dfs((1<<m)-1, k, [])
    if res is None:
        return None
    vals = [A[active[v]] for v in res]
    # verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals
>>>>>>> REPLACE
