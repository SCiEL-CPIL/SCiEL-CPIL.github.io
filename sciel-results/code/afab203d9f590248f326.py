def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 12
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j  # term index
            U[i * m + j, r] = 1
            for k in range(p):
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
