<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Attempt the known best rank 28 decomposition
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, tensor = verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {'U': U, 'V': V, 'W': W}
    except Exception as e:
        pass

    # Fallback: Schoolbook (Rank 30)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if valid:
        return {'U': U, 'V': V, 'W': W}
    
    # Should not happen
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try to find a rank-27 decomposition
    try:
        result_27 = try_rank_27(n, m, p)
        if result_27 is not None:
            U, V, W = result_27
            valid, _ = verify_decomposition_fast(U, V, W, n, m, p)
            if valid:
                return {'U': U, 'V': V, 'W': W}
    except Exception:
        pass
    
    # Attempt the known best rank 28 decomposition
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, tensor = verify_decomposition_fast(U, V, W, n, m, p)
        if valid:
            return {'U': U, 'V': V, 'W': W}
    except Exception as e:
        pass

    # Fallback: Schoolbook (Rank 30)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    valid, _ = verify_decomposition_fast(U, V, W, n, m, p)
    if valid:
        return {'U': U, 'V': V, 'W': W}
    
    # Should not happen
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
>>>>>>> REPLACE
