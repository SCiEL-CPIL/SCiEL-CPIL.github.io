def construct():
    n, m, p = 3, 4, 7
    
    # We'll use the splitting strategy: rank(n,m,p) <= n by splitting n into n unit parts
    # Each part is rank(1,m,p) = 1
    
    rank = n  # = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        # Part i: rank-1 decomposition for the (1,m,p) sub-problem at row i
        for j in range(m):
            U[i * m + j][i] = 1  # U[i*m+j, i] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][i] = 1  # V[j*p+k, i] = 1
        for k in range(p):
            W[k * n + i][i] = 1  # W[k*n+i, i] = 1
    
    return {"U": U, "V": V, "W": W}
