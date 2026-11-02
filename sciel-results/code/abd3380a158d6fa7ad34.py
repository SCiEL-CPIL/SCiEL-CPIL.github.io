<<<<<<< SEARCH
    # Verify the rank-10 decomposition
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If rank 10 doesn't work, try rank 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            for i in range(n):
                U[i * m + j0][c] = 1
                W[k0 * n + i][c] = 1
            V[j0 * p + k0][c] = 1
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
=======
    # Verify the rank-10 decomposition
    if verify_mm(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 50)
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
