def _aggressive_reduction(U, V, W, n, m, p, max_time=120):
    start_time = time.time()
    deadline = start_time + max_time
    rank = U.shape[1]
    
    # First pass: try removing each single term
    for r in range(rank):
        if time.time() > deadline - 10:
            break
        result = _try_remove_single_term(U, V, W, n, m, p, r)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
    
    # Second pass: try removing pairs of terms
    for r1 in range(rank):
        if time.time() > deadline - 10:
            break
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline - 10:
                break
            result = _try_remove_two_terms(U, V, W, n, m, p, r1, r2)
            if result is not None:
                U, V, W = result
                rank = U.shape[1]
                break  # Restart pair search with new rank
    
    # Third pass: Null-space elimination
    if time.time() < deadline - 20:
        reduced = _null_space_elimination(U, V, W, n, m, p, max_time=min(60, deadline - time.time() - 5))
        if reduced is not None:
            U, V, W = reduced
            rank = U.shape[1]
            
    return U, V, W
