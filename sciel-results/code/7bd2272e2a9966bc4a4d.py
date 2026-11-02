def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = p * n  # 24
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            # U[i*m + j, c] = 1 for all j
            for j in range(m):
                U[i * m + j, c] = 1
            # V[j*p + k2, c] = 1 for all j, k2
            for j in range(m):
                for k2 in range(p):
                    V[j * p + k2, c] = 1
            # W[k*n + i2, c] = 1 if i2 == i, else 0
            W[k * n + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
