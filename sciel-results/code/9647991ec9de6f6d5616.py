def construct():
    n, m, p = 2, 3, 4
    rank = m * p  # 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 12 x 12
    W = [[0] * rank for _ in range(p * n)]  # 8 x 12
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U[i*m + j, r] = 1 for i in range(n)
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            # W[k*n + i, r] = 1 for i in range(n)
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
