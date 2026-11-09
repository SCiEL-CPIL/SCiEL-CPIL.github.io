def _rank3_solution():
    n, m, p = 3, 3, 6
    rank = 3
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c in range(3):
        # U[i*m+j, c] = 1 if i == c
        for j in range(m):
            U[c*m+j, c] = 1.0
        
        # V[j*p+k, c] = 1 if k == c
        for j in range(m):
            V[j*p+c, c] = 1.0
        
        # W[k*n+i, c] = 1 if i == c
        for k in range(p):
            W[k*n+c, c] = 1.0
    
    return U, V, W
