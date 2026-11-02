def construct():
    n, m, p = DIMENSIONS
    rank = p  # 9
    U = [[0] * rank for _ in range(n * m)]  # 6 x 9
    V = [[0] * rank for _ in range(m * p)]  # 27 x 9
    W = [[0] * rank for _ in range(p * n)]  # 18 x 9
    
    for k in range(p):
        c = k  # term index
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1
        for j in range(m):
            V[j * p + k][c] = 1
        for i in range(n):
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
