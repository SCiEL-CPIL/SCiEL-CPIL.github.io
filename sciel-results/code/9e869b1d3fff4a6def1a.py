def _local_search(adj, rng, all_edges, deadline):
    best_count = _count_edges(adj)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing edge
        for u, v in all_edges:
            if time.time() > deadline:
                return adj, best_count
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        
        if improved:
            best_count = _count_edges(adj)
            continue
        
        # 2. Swaps
        existing_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
        if not existing_edges:
            break
            
        # Try a few random swaps
        for _ in range(10):
            if time.time() > deadline:
                return adj, best_count
                
            u_rem, v_rem = rng.choice(existing_edges)
            adj_backup = adj[:]
            
            _remove_edge(adj, u_rem, v_rem)
            
            # Try to add 2 edges
            candidates = [e for e in all_edges if e != (u_rem, v_rem) and e != (v_rem, u_rem) and not ((adj[e[0]] >> e[1]) & 1)]
            if not candidates:
                adj = adj_backup
                continue
            rng.shuffle(candidates)
            
            added_count = 0
            for u_add, v_add in candidates[:20]:
                if _creates_c6(adj, u_add, v_add):
                    continue
                _add_edge(adj, u_add, v_add)
                added_count += 1
                if added_count == 2:
                    break
            
            if added_count > 0:
                best_count = _count_edges(adj)
                improved = True
                break # Restart the main loop with new structure
            else:
                adj = adj_backup
                # If we failed to add 2, maybe try adding just 1?
                # Or just revert and try another swap.
                # Let's try adding 1 to shuffle.
                for u_add, v_add in candidates[:20]:
                    if _creates_c6(adj, u_add, v_add):
                        continue
                    _add_edge(adj, u_add, v_add)
                    improved = True
                    break
                if not improved:
                    adj = adj_backup

    return adj, best_count
