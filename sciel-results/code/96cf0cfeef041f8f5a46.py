def construct():
    n, m, p = 3, 5, 5
    rank = n * m  # 15
    U = [[0] * rank for _ in range(n * m)]  # 15 x 15
    V = [[0] * rank for _ in range(m * p)]  # 25 x 15
    W = [[0] * rank for _ in range(p * n)]  # 15 x 15
    
    for i in range(n):
        for j in range(m):
            col = i * m + j
            U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
