<<<<<<< SEARCH
def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    # Schoolbook fallback (always valid)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook

    # Known lower bound: rank >= max(n*m, m*p, p*n) = 20
    # Try ranks from 20 upward
    for rank in range(20, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break
        # Multiple seeds per rank
        for seed in range(50):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
    return best_result
=======
def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    
    # Try Rank 1 first: The all-ones tensor has rank 1.
    # If the target is the all-ones tensor (as per seed comment), this is optimal.
    try:
        U1 = [[1.0] for _ in range(n * m)]
        V1 = [[1.0] for _ in range(m * p)]
        W1 = [[1.0] for _ in range(p * n)]
        if _verify(U1, V1, W1):
            return {"U": U1, "V": V1, "W": W1}
    except Exception:
        pass

    # Schoolbook fallback (always valid)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook

    # Try ranks from 2 upward up to schoolbook rank
    # In case rank 1 fails (if target is not all-ones), we might find a better rank than 60
    # But given the comment, rank 1 should work.
    # We'll keep the search from 20 as before, but now we've handled the trivial case.
    for rank in range(2, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break
        # Multiple seeds per rank
        for seed in range(50):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
    return best_result
>>>>>>> REPLACE
