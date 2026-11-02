def construct():
    n, m, p = 2, 2, 15
    rank = n * m  # = 4
    U = [[0] * rank for _ in range(n * m)]  # 4 x 4
    V = [[0] * rank for _ in range(m * p)]  # 30 x 4
    W = [[0] * rank for _ in range(p * n)]  # 30 x 4
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U[i*m+j, col] = 1
            U[i*m+j][col] = 1
            # V[j*p+k, col] = 1 for all k
            for k in range(p):
                V[j*p+k][col] = 1
            # W[k*n+i, col] = 1 for all k
            for k in range(p):
                W[k*n+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
