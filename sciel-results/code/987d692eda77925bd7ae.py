def construct():
    n, m, p = 2, 3, 6
    rank = m * p  # 18
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U[i*m+j] = 1 for i=0,1
            for i in range(n):
                U[i * m + j][col] = 1
            # V[j*p+k] = 1
            V[j * p + k][col] = 1
            # W[k*n+i] = 1 for i=0,1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
