def construct():
    n, m, p = DIMENSIONS
    N_t = n * m
    M_t = m * p
    P_t = p * n
    
    # Schoolbook decomposition
    r = n * m * p
    U = np.zeros((N_t, r))
    V = np.zeros((M_t, r))
    W = np.zeros((P_t, r))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[_t0(i, m, j), col] = 1.0
                V[_t1(j, p, k), col] = 1.0
                W[_t2(k, n, i), col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
