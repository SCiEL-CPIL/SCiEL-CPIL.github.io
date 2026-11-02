def construct():
    n, m, p = 2, 5, 5
    R = 10
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # For i=0, use ranks 0-4
    for j in range(m):
        U[j, j] = 1  # U[0*5+j, j] = 1
        for k in range(p):
            V[j*p+k, j] = 1  # V[j*5+k, j] = 1
        for k in range(p):
            W[k*n+0, j] = 1  # W[k*2+0, j] = 1
    
    # For i=1, use ranks 5-9
    for j in range(m):
        U[m+j, m+j] = 1  # U[5+j, 5+j] = 1
        for k in range(p):
            V[j*p+k, m+j] = 1  # V[j*5+k, 5+j] = 1
        for k in range(p):
            W[k*n+1, m+j] = 1  # W[k*2+1, 5+j] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
