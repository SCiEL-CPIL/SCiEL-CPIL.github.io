<<<<<<< SEARCH
def _try_reduce_rank(U, V, W, n, m, p, max_reductions, time_budget):
    """Iteratively try to reduce the rank by eliminating terms."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    current_rank = U_cur.shape[1]

    for _ in range(max_reductions):
        if time.time() - start_time > time_budget:
            break
        U_cur, V_cur, W_cur = _try_eliminate_one_term(U_cur, V_cur, W_cur, n, m, p)
        if U_cur.shape[1] == current_rank:
            break
        current_rank = U_cur.shape[1]

    return U_cur, V_cur, W_cur
=======
def _try_reduce_rank(U, V, W, n, m, p, max_reductions, time_budget):
    """Iteratively try to reduce the rank by eliminating terms."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    current_rank = U_cur.shape[1]

    for _ in range(max_reductions):
        if time.time() - start_time > time_budget:
            break
        
        # Try single term elimination first
        U_new, V_new, W_new = _try_eliminate_one_term(U_cur, V_cur, W_cur, n, m, p)
        
        if U_new.shape[1] < current_rank:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            current_rank = U_cur.shape[1]
            continue
        
        # If single didn't work, try multi-term elimination
        if time.time() - start_time < time_budget - 5:
            U_new, V_new, W_new = _try_eliminate_multi_terms(U_cur, V_cur, W_cur, n, m, p, max_terms=2)
            if U_new.shape[1] < current_rank:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                current_rank = U_cur.shape[1]
                continue
        
        # No progress
        break

    return U_cur, V_cur, W_cur
>>>>>>> REPLACE

