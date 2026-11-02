def construct():
    n, m, p = DIMENSIONS
    rank = p  # 13
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][k] = 1
                V[j * p + k][k] = 1
                W[k * n + i][k] = 1
    
    return {"U": U, "V": V, "W": W}
