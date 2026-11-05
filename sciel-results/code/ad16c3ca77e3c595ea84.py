def construct():
    n, m, p = 3, 4, 4
    rank = m * p  # 16
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 16
    V = [[0] * rank for _ in range(m * p)]  # 16 x 16
    W = [[0] * rank for _ in range(p * n)]  # 12 x 16
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
