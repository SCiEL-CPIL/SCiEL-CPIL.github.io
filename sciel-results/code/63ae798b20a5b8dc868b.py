def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is identity: V[j*p+k, j*p+k] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][j * p + k] = 1
    
    # U[i*m+j, r] = 1 for r in {j*p, j*p+1, ..., j*p+p-1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U[i * m + j][r] = 1
    
    # W[k*n+i, r] = 1 for r in {k, p+k, 2p+k, ..., (m-1)p+k}
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = j * p + k
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
