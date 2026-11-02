def construct():
    n, m, p = 2, 3, 6
    rank = 6
    U = [[0]*rank for _ in range(n*m)]  # 6 x 6
    V = [[0]*rank for _ in range(m*p)]  # 18 x 6
    W = [[0]*rank for _ in range(p*n)]  # 12 x 6
    
    for j in range(m):
        for i in range(n):
            r = j * n + i  # r = 2j + i
            # U: nonzero at (i,j) -> index i*m + j
            U[i*m + j][r] = 1
            # V: nonzero at (j,k) for all k -> indices j*p + k
            for k in range(p):
                V[j*p + k][r] = 1
            # W: nonzero at (k,i) for all k -> indices k*n + i
            for k in range(p):
                W[k*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
