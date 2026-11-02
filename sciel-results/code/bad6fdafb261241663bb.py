def construct():
    n, m, p = 4, 4, 9
    rank = 16
    
    U = [[0]*rank for _ in range(n*m)]  # 16x16
    V = [[0]*rank for _ in range(m*p)]  # 36x16
    W = [[0]*rank for _ in range(p*n)]  # 36x16
    
    # U is identity: U[i*m+j, i*m+j] = 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][i*m+j] = 1
    
    # V[j*p+k, i*m+j] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                V[j*p+k][i*m+j] = 1
    
    # W[k*n+i, i*m+j] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k*n+i][i*m+j] = 1
    
    return {"U": U, "V": V, "W": W}
