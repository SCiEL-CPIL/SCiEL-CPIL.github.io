def construct():
    n, m, p = DIMENSIONS
    # Construct a decomposition of rank n*p
    # This works by assigning one term to each (i, k) pair
    r = n * p
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            # U[i, j, c] = 1 for all j
            for j in range(m):
                U[i * m + j, c] = 1.0
            # V[j, k, c] = 1 for all j
            for j in range(m):
                V[j * p + k, c] = 1.0
            # W[k, i, c] = 1
            W[k * n + i, c] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
