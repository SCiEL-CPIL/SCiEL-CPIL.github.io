<<<<<<< SEARCH
def _local_search(a, b, p, current_cost, max_iters=2000, time_limit=5.0):
    """
    Steepest-descent local search: repeatedly find the best improving swap.
    Uses O(n^2) delta computation per iteration.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Compute all swap deltas for this permutation
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # Local minimum
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
    
    return p, cost
=======
def _local_search(a, b, p, current_cost, max_iters=500, time_limit=5.0):
    """
    Steepest-descent local search using vectorized swap delta computation.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all swap deltas vectorized
        D = _compute_all_swap_deltas(a, b, p)
        
        # Find best improving swap (upper triangle only)
        D_upper = np.triu(D, k=1)
        best_delta, best_idx = np.unravel_index(np.argmin(D_upper), D_upper.shape)
        
        if best_delta >= 0:
            break  # Local minimum
        
        i, j = int(best_idx[0]), int(best_idx[1])
        p[i], p[j] = p[j], p[i]
        cost += best_delta
    
    return p, cost
>>>>>>> REPLACE

