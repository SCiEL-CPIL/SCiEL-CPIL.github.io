    best_n, best_edges = float('inf'), None
    
    # Try LUW constructions with varying d
    q = k
    # Original d estimate
    d_max = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    for d in range(3, d_max + 1):
        if time.time() - t0 > _DEADLINE - _SAFETY:
            break
        n_candidate, edges_candidate = _lazebnik_construct_d(q, d)
        # Build adjacency for check
        adj = [set() for _ in range(n_candidate)]
        for u, v in edges_candidate:
            adj[u].add(v)
            adj[v].add(u)
        adj = [list(s) for s in adj]
        
        if _girth_ok(adj, n_candidate, g):
            if n_candidate < best_n:
                best_n = n_candidate
                best_edges = edges_candidate
            # Since n increases with d, the first valid d (smallest) gives the smallest n
            # So we can break
            break

    # If no LUW graph found (should not happen for d_max), use original as fallback
    if best_n == float('inf'):
         best_n, best_edges = _lazebnik_construct(k, g)
