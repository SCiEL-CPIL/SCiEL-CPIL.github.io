<<<<<<< SEARCH
    # 3. General Case: LUW
    return luw_construct(k, g)
=======
    # 3. General Case: LUW
    n_luw, edges_luw = luw_construct(k, g)
    
    # 4. Try to find smaller graph via random construction + local search
    if time.time() < deadline - 20:
        if g % 2 == 1:
            r = (g - 1) // 2
            moore = 1 + k * sum(k**i for i in range(r))
        else:
            r = g // 2
            moore = 2 * sum(k**i for i in range(r))
        
        best_n = n_luw
        best_edges = edges_luw
        
        # Try a range of n values from Moore bound upward
        max_trial_n = min(n_luw, moore + 300)
        for trial_n in range(max(moore, 2 * k), max_trial_n):
            if time.time() > deadline - 15:
                break
            if trial_n >= best_n:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 15:
                    break
                
                adj = _random_k_regular(trial_n, k, rng)
                if adj is None:
                    continue
                
                edges = []
                for u in range(trial_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                
                girth = compute_girth(trial_n, edges, min(g + 1, 20))
                
                if girth >= g:
                    best_n = trial_n
                    best_edges = edges
                    found = True
                    break
                
                if girth >= g - 2:
                    new_girth = try_edge_switch(adj, trial_n, k, g, rng, max_iterations=10000)
                    if new_girth >= g:
                        new_edges = []
                        for u in range(trial_n):
                            for v in adj[u]:
                                if u < v:
                                    new_edges.append((u, v))
                        if verify_girth(trial_n, new_edges, g):
                            best_n = trial_n
                            best_edges = new_edges
                            found = True
                            break
        
            if found:
                # Continue searching for even smaller
                continue
        
        if best_n < n_luw:
            return best_n, best_edges
    
    return n_luw, edges_luw
>>>>>>> REPLACE
