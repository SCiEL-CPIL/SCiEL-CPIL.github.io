<<<<<<< SEARCH
    # Run multiple random restarts with simulated annealing
    restarts = 0
    while time.time() < deadline - 10:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        # Local search to fill in missing edges
        count = local_search(adj, count)
        
        # Simulated annealing to escape local optima
        count = simulated_annealing(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
                    
        restarts += 1
        
        # If we are running out of time, break early
        if time.time() > deadline - 15:
            break

    return best_edges
=======
    # Run multiple random restarts with simulated annealing
    restarts = 0
    while time.time() < deadline - 5:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        # Local search to fill in missing edges
        count = local_search(adj, count)
        
        # 2-opt swaps to improve further
        improved, count = two_opt_swap(adj, count)
        while improved and time.time() < deadline - 5:
            improved, count = two_opt_swap(adj, count)
        
        # Simulated annealing to escape local optima
        count = simulated_annealing(adj, count)
        
        # Final polish: another round of local search + 2-opt
        count = local_search(adj, count)
        improved, count = two_opt_swap(adj, count)
        while improved and time.time() < deadline - 5:
            improved, count = two_opt_swap(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
                    
        restarts += 1
        
        # If we are running out of time, break early
        if time.time() > deadline - 8:
            break

    return best_edges
>>>>>>> REPLACE
