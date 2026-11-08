def _try_reduce_rank(U, V, W, max_time=100):
    start = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    active = list(range(best_rank))
    
    # Iterate until no more reduction or time out
    # Since removing a column changes the set, we restart or iterate carefully.
    # Simple greedy: check each active column against others.
    # To avoid index shifting issues, we just build the final matrices at the end.
    
    current_active = list(range(best_rank))
    
    while True:
        if time.time() - start > max_time:
            break
        reduced = False
        new_active = []
        for idx in current_active:
            # Check if column idx is in span of others in current_active
            others = [i for i in current_active if i != idx]
            if not others:
                new_active.append(idx)
                continue
            
            U_others = best_U[:, others]
            V_others = best_V[:, others]
            W_others = best_W[:, others]
            
            A_mat = np.vstack([U_others, V_others, W_others])
            b_vec = np.concatenate([best_U[:, idx], best_V[:, idx], best_W[:, idx]])
            
            try:
                alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                A_alpha = A_mat @ alpha
                if np.allclose(A_alpha, b_vec, atol=1e-10):
                    reduced = True
                    # Do not add idx to new_active
                else:
                    new_active.append(idx)
            except:
                new_active.append(idx)
        
        current_active = new_active
        if not reduced:
            break
            
    if current_active:
        best_U = best_U[:, current_active]
        best_V = best_V[:, current_active]
        best_W = best_W[:, current_active]
    else:
        # Edge case: all reduced? Unlikely.
        pass
        
    return best_U, best_V, best_W, len(current_active)
