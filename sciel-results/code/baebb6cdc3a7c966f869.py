def construct():
    import time
    start_time = time.time()
    time_limit = 140.0

    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem_adj = adj[:]
        cliques = []
        active_mask = (1 << n) - 1
        
        def find_large_clique(rem_adj, n, active_mask, deadline):
            """Find a large clique in the subgraph defined by rem_adj and active_mask."""
            # Get vertices with degree > 0 in rem_adj
            verts = []
            m = active_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if rem_adj[idx]:
                    verts.append(idx)
                m ^= lsb
            
            if not verts:
                return []
            
            # Sort by degree in rem_adj (descending)
            def deg(v):
                return (rem_adj[v] & active_mask).bit_count()
            verts.sort(key=deg, reverse=True)
            
            # Try to build a clique starting from the highest-degree vertex
            best_clique = []
            for start_v in verts[:min(10, len(verts))]:
                # Build clique greedily from start_v
                clique = [start_v]
                candidates = rem_adj[start_v] & active_mask
                candidates &= ~(1 << start_v)
                
                while candidates:
                    # Pick vertex with highest degree in candidates
                    best_v = -1
                    best_deg = -1
                    cm = candidates
                    while cm:
                        lsb = cm & -cm
                        v = lsb.bit_length() - 1
                        d = (rem_adj[v] & candidates).bit_count()
                        if d > best_deg:
                            best_deg = d
                            best_v = v
                        cm ^= lsb
                    
                    if best_v == -1:
                        break
                    
                    # Check if best_v is connected to all in clique
                    ok = True
                    for u in clique:
                        if not (rem_adj[best_v] >> u) & 1:
                            ok = False
                            break
                    if not ok:
                        candidates &= ~(1 << best_v)
                        continue
                    
                    clique.append(best_v)
                    # Remove best_v from candidates and its non-neighbors
                    new_candidates = candidates & rem_adj[best_v]
                    new_candidates &= ~(1 << best_v)
                    candidates = new_candidates
                
                if len(clique) > len(best_clique):
                    best_clique = clique
                    
                    # Early termination if we found a large clique
                    if len(best_clique) >= 8:
                        break
                
                if time.time() > deadline:
                    break
            
            if not best_clique:
                # Fallback: find any edge
                for u in verts:
                    neigh = rem_adj[u] & active_mask
                    if neigh:
                        lsb = neigh & -neigh
                        v = lsb.bit_length() - 1
                        best_clique = [u, v]
                        break
            
            return best_clique

        deadline = start_time + time_limit
        
        while True:
            if time.time() > deadline:
                break
            
            clique = find_large_clique(rem_adj, n, active_mask, deadline)
            
            if not clique:
                break
            
            cliques.append(clique)
            
            # Remove edges in clique from rem_adj
            for u in clique:
                for v in clique:
                    if u != v:
                        rem_adj[u] &= ~(1 << v)
            
            # Update active_mask
            active_mask = 0
            for i in range(n):
                if rem_adj[i]:
                    active_mask |= (1 << i)
            
            if not active_mask:
                break
        
        # Fallback: cover remaining edges as pairs
        if active_mask:
            for u in range(n):
                neigh = rem_adj[u] & active_mask
                while neigh:
                    lsb = neigh & -neigh
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                    neigh ^= lsb
        
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
