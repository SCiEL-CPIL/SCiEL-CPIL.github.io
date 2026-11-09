def construct():
    n, m, p = 3, 4, 8
    rank = n * m  # 12
    U = [[0] * rank for _ in range(n * m)]  # 12 x 12
    V = [[0] * rank for _ in range(m * p)]  # 32 x 12
    W = [[0] * rank for _ in range(p * n)]  # 24 x 12
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
