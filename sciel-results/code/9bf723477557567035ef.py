def construct():
    n, m, p = 4, 4, 5
    rank = p  # 5
    
    U = np.ones((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        # V[j*p+r, r] = 1 for all j
        for j in range(m):
            V[j*p + r, r] = 1
        # W[r*n+i, r] = 1 for all i
        for i in range(n):
            W[r*n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
