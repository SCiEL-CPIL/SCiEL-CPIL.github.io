def construct():
    n, m, p = 4, 5, 6
    
    # Try Strassen-based decomposition first
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Strassen 4x5x6
    try:
        if _time_left() > 30.0:
            U, V, W = _build_strassen_4x5x6_v2()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try iterative refinement with projected Strassen initialization
    try:
        if _time_left() > 20.0:
            # Use the best Strassen decomposition found so far (likely rank 108)
            U_s, V_s, W_s = best_U, best_V, best_W
            R_s = U_s.shape[1]
            
            # Try reducing rank by projecting
            for R in range(R_s - 2, 59, -2):
                if _time_left() < 15.0:
                    break
                
                # Random projection
                np.random.seed(100 + R)
                P = np.random.randn(R_s, R)
                U_proj = U_s @ P
                V_proj = V_s @ P
                W_proj = W_s @ P
                
                cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=50, 
                                             U_init=U_proj, V_init=V_proj, W_init=W_proj)
                if cand is not None:
                    U, V, W = cand
                    if _verify_decomp(U, V, W, n, m, p, 1e-9):
                        if U.shape[1] < best_rank:
                            best_rank = U.shape[1]
                            best_U, best_V, best_W = U, V, W
                            # Break if we hit a very low rank? Or continue to try lower?
                            # Let's continue to try lower ranks to find the minimum.
    except Exception:
        pass
    
    # Local column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=10, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pairwise column-removal refinement
    try:
        if _time_left() > 8.0:
            cand = _pairwise_reduce(best_U, best_V, best_W, n, m, p, max_iters=5, budget_s=10.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
