<<<<<<< SEARCH
    # Start with 32-vertex coloring
    adj = make_adj_32()
    best_n = 32
    best_colors = adj_to_colors(adj, best_n)
    
    # Try to expand
    for target_n in range(33, 51):
        if time_left() < 10:
            break
        
        n = len(adj)
        found = False
        
        # Try backtracking first (exact, but may be slow)
        if time_left() > 5:
            bt_limit = min(5.0, time_left() - 5)
            assignment = add_vertex_backtrack(adj, n, bt_limit)
            if assignment is not None:
                # Success
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
                break
        
        # If backtracking didn't work, try SA
        if not found and time_left() > 5:
            sa_limit = min(10.0, time_left() - 5)
            assignment = add_vertex_sa(adj, n, sa_limit)
            if assignment is not None:
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
        
        if not found:
            break
    
    # Final validation
    adj_final = colors_to_adj(best_colors, best_n)
    if not is_valid(adj_final, best_n):
        return {'n': 32, 'colors': adj_to_colors(make_adj_32(), 32)}
    
    return {'n': best_n, 'colors': best_colors}
=======
    # Start with 32-vertex coloring
    adj = make_adj_32()
    best_n = 32
    best_colors = adj_to_colors(adj, best_n)
    
    # Try to expand
    for target_n in range(33, 51):
        if time_left() < 5:
            break
        
        n = len(adj)
        found = False
        
        # Try backtracking first (exact, but may be slow)
        if time_left() > 3:
            bt_limit = min(3.0, time_left() - 3)
            assignment = add_vertex_backtrack(adj, n, bt_limit)
            if assignment is not None:
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
                break
        
        # If backtracking didn't work, try SA
        if not found and time_left() > 3:
            sa_limit = min(20.0, time_left() - 3)
            assignment = add_vertex_sa(adj, n, sa_limit)
            if assignment is not None:
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
        
        if not found:
            # Try a few more times with different random seeds before giving up
            retry_count = 0
            while not found and retry_count < 3 and time_left() > 5:
                sa_limit = min(10.0, time_left() - 5)
                assignment = add_vertex_sa(adj, n, sa_limit)
                if assignment is not None:
                    new_n = n + 1
                    new_adj = [row[:] for row in adj]
                    for _ in range(new_n):
                        new_adj.append([-1] * new_n)
                    for i in range(n):
                        new_adj[i][n] = new_adj[n][i] = assignment[i]
                    new_adj[n][n] = -1
                    adj = new_adj
                    best_n = new_n
                    best_colors = adj_to_colors(adj, best_n)
                    found = True
                retry_count += 1
            
            if not found:
                break
    
    # Final validation
    adj_final = colors_to_adj(best_colors, best_n)
    if not is_valid(adj_final, best_n):
        return {'n': 32, 'colors': adj_to_colors(make_adj_32(), 32)}
    
    return {'n': best_n, 'colors': best_colors}
>>>>>>> REPLACE
