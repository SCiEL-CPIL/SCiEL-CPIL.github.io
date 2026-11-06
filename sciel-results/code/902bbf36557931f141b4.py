<<<<<<< SEARCH
    # Local search: try all single-edge modifications systematically
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try all possible edge additions
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if current_adj[u] & (1 << v):
                    continue
                if not can_add_edge(current_adj, u, v):
                    continue
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                add_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
            if time.time() > deadline - 2:
                break
        
        # Try all possible edge removals (less likely to help but can escape local minima)
        if time.time() > deadline - 2:
            break
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if not (current_adj[u] & (1 << v)):
                    continue
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                remove_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
            if time.time() > deadline - 2:
                break
=======
    # Local search: try all single-edge modifications systematically
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try all possible edge additions
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if current_adj[u] & (1 << v):
                    continue
                if not can_add_edge(current_adj, u, v):
                    continue
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                add_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
            if time.time() > deadline - 2:
                break
        
        # Try all possible edge removals (less likely to help but can escape local minima)
        if time.time() > deadline - 2:
            break
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if not (current_adj[u] & (1 << v)):
                    continue
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                remove_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
            if time.time() > deadline - 2:
                break
        
        # Second pass: try 2-edge flips (add one, remove another) for escaping local minima
        if time.time() > deadline - 3:
            break
        for u1 in range(n):
            for v1 in range(u1 + 1, n):
                if time.time() > deadline - 3:
                    break
                if current_adj[u1] & (1 << v1):
                    continue
                if not can_add_edge(current_adj, u1, v1):
                    continue
                old_u1 = current_adj[u1]
                old_v1 = current_adj[v1]
                add_edge(current_adj, u1, v1)
                
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if time.time() > deadline - 3:
                            break
                        if not (current_adj[u2] & (1 << v2)):
                            continue
                        if u2 == u1 and v2 == v1:
                            continue
                        old_u2 = current_adj[u2]
                        old_v2 = current_adj[v2]
                        remove_edge(current_adj, u2, v2)
                        new_score = evaluate(current_adj)
                        
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_adj = current_adj[:]
                            improved = True
                        else:
                            current_adj[u2] = old_u2
                            current_adj[v2] = old_v2
                    if time.time() > deadline - 3:
                        break
                # Restore the added edge if we didn't improve
                current_adj[u1] = old_u1
                current_adj[v1] = old_v1
            if time.time() > deadline - 3:
                break
>>>>>>> REPLACE
