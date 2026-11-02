def construct():
    n,m,p = 2,5,6
    # Build T
    T = np.zeros((p*n, n*m, m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = k*n + i; y = i*m + j; z = j*p + k
                T[x,y,z] = 1.0
    # Block solution (rank 54)
    Ub, Vb, Wb = _build_block_solution(n,m,p)
    assert _verify(Ub, Vb, Wb, n,m,p), "block solution invalid"
    best = (Ub, Vb, Wb)
    best_rank = 54
    rng = np.random.RandomState(12345)
    # Try to reduce rank
    for r in range(53, 29, -1):
        if _left() < 25:
            break
        found = False
        # Init 1: from block solution (zero out random columns to r, then ALS)
        # Actually block is 54 cols; to get rank r, take first r cols? Not meaningful.
        # Better: random restarts
        n_restarts = 6 if r >= 45 else 4
        for rs in range(n_restarts):
            if _left() < 15:
                break
            U0 = rng.randn(n*m, r)
            V0 = rng.randn(m*p, r)
            W0 = rng.randn(p*n, r)
            U1, V1, W1 = _als(U0, V0, W0, T, max_iter=300)
            Us = _snap_half(U1); Vs = _snap_half(V1); Ws = _snap_half(W1)
            if Us is not None and Vs is not None and Ws is not None:
                if _verify(Us, Vs, Ws, n,m,p):
                    best = (Us, Vs, Ws); best_rank = r; found = True
                    break
        if found:
            break
    # Also try: start from block solution, reduce columns and re-optimize
    # (optional, if time)
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
