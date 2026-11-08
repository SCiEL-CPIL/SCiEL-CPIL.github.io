def construct():
    n, m, p = 2, 3, 7
    rank = m * p  # 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            r = j0 * p + k0
            for i in range(n):
                U[i * m + j0][r] = 1
                W[k0 * n + i][r] = 1
            V[j0 * p + k0][r] = 1
    
    return {"U": U, "V": V, "W": W}
