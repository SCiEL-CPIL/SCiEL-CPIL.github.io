def _greedy_construct(n, k, g):
    if (n * k) % 2 != 0:
        return None
        
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
            
    random.shuffle(stubs)
    
    max_depth = g - 2
    failed_picks = 0
    max_failed_picks = 10000
    
    while len(stubs) >= 2:
        if failed_picks > max_failed_picks:
            return None # Stuck
            
        idx_u = random.randint(0, len(stubs) - 1)
        idx_v = random.randint(0, len(stubs) - 1)
        
        if idx_u == idx_v:
            continue # Don't count as failure? Or do?
            
        u = stubs[idx_u]
        v = stubs[idx_v]
        
        if u == v:
            failed_picks += 1
            continue
            
        if (min(u, v), max(u, v)) in edge_set:
            failed_picks += 1
            continue
            
        if _is_too_close(u, v, adj, max_depth):
            failed_picks += 1
            continue
            
        # Accept edge
        edge_set.add((min(u, v), max(u, v)))
        adj[u].append(v)
        adj[v].append(u)
        
        failed_picks = 0 # Reset on success
        
        # Remove stubs
        if idx_u > idx_v:
            idx_u, idx_v = idx_v, idx_u
            
        stubs[idx_v] = stubs[-1]
        stubs.pop()
        
        stubs[idx_u] = stubs[-1]
        stubs.pop()
        
    return n, list(edge_set)
