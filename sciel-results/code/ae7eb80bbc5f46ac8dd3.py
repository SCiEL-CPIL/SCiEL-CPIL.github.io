def _try_local_search_v2(U, V, W, n, m, p, max_iterations=200, time_limit=60):
    """
    More aggressive local search: try replacing small groups of terms.
    """
    start_time = time.time()
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    best_rank = rank
    
    # Try removing 1, 2, or 3 terms and see if we can compensate
    for num_remove in [1, 2, 3]:
        if time.time() - start_time > time_limit:
            break
        
        # Try all combinations of num_remove terms to remove
        from itertools import combinations
        for remove_indices in combinations(range(rank), num_remove):
            if time.time() - start_time > time_limit:
                break
            
            mask = np.ones(rank, dtype=bool)
            mask[list(remove_indices)] = False
            new_rank = rank - num_remove
            
            U_test = U_arr[:, mask]
            V_test = V_arr[:, mask]
            W_test = W_arr[:, mask]
            
            # Check if still valid
            if _random_eval_check(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, num_trials=3):
                best_rank = new_rank
                best_U = U_test.copy()
                best_V = V_test.copy()
                best_W = W_test.copy()
                break  # Found improvement, restart with new base
    
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None
