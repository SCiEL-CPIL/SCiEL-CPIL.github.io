<<<<<<< SEARCH
    # Strategy 2: Try random init + ALS for ranks lower than current best
    for r in range(best_rank - 1, max(10, best_rank // 2), -1):
        if time.time() > deadline - 15:
            break
        time_per_attempt = max(1, (deadline - 15 - time.time()) / 3)
        result = _try_rank_random(r, n, m, p, n_restarts=5, time_limit=time_per_attempt)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # Final polish of best result
    U, V, W = best_result
    if time.time() < deadline - 5:
        polished = _als_decompose(U, V, W, n, m, p, max_iter=200, tol=1e-12, time_limit=5)
        if polished is not None:
            U, V, W = polished
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-5):
        # Fall back to schoolbook
        U, V, W = _schoolbook()
=======
    # Strategy 2: Try random init + ALS for ranks lower than current best
    for r in range(best_rank - 1, max(10, best_rank // 2), -1):
        if time.time() > deadline - 15:
            break
        time_per_attempt = max(1, (deadline - 15 - time.time()) / 4)
        result = _try_rank_random(r, n, m, p, n_restarts=8, time_limit=time_per_attempt)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # Final polish of best result with high precision
    U, V, W = best_result
    if time.time() < deadline - 5:
        polished = _als_decompose(U, V, W, n, m, p, max_iter=300, tol=1e-12, time_limit=5)
        if polished is not None:
            U, V, W = polished
        # Additional polish pass with even tighter tolerance
        if time.time() < deadline - 2:
            polished2 = _als_decompose(U, V, W, n, m, p, max_iter=100, tol=1e-14, time_limit=2)
            if polished2 is not None:
                U, V, W = polished2
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-5):
        # Fall back to schoolbook
        U, V, W = _schoolbook()
>>>>>>> REPLACE
