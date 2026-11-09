def check_new_vertex_fast(adj, new_neighbors_mask, n_existing):
    # adj: list of masks for 0..n_existing-1
    # new_neighbors_mask: mask of neighbors of new vertex in 0..n_existing-1
    
    # We need to check all T subset of {0..n_existing-1} with |T| >= 5
    # such that S = T U {new} is regular.
    
    # Optimization:
    # Iterate over T.
    # For each T, compute deg_new = popcount(new_neighbors_mask & T_mask)
    # For each u in T, deg_u = popcount(adj[u] & T_mask) + (1 if u in new_neighbors_mask)
    # Check if all deg_u == deg_new.
    
    # To speed up, we can iterate over T using combinations.
    # For n_existing up to 20, this is heavy.
    # But we do this incrementally.
    
    # If n_existing is large, maybe we can use a heuristic?
    # No, we need correctness.
    
    # Let's use a helper for popcount.
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
        
    # Precompute neighbor masks for existing vertices?
    # adj is already masks.
    
    # Iterate
    for t_size in range(5, n_existing + 1):
        for T in combinations(range(n_existing), t_size):
            T_mask = 0
            for v in T:
                T_mask |= (1 << v)
            
            deg_new = popcount(new_neighbors_mask & T_mask)
            
            # Check all u in T
            ok = True
            for u in T:
                deg_u = popcount(adj[u] & T_mask)
                if new_neighbors_mask & (1 << u):
                    deg_u += 1
                if deg_u != deg_new:
                    ok = False
                    break
            
            if ok:
                return False # Found regular subset
    
    return True
