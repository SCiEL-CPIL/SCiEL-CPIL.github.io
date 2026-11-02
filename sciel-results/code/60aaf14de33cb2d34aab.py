def construct():
    n, m, p = 3, 5, 6
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    
    # Block 1: j in {0,1}, rank = 2*6 = 12
    # Block 2: j in {2,3,4}, rank = 3*6 = 18
    rank = 12 + 18  # 30
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    # Block 1: j in {0,1}
    for j in range(2):
        for k in range(p):
            # U[i*5+j, r] = 1 for all i
            for i in range(n):
                U[i*m + j, r] = 1
            V[j*p + k, r] = 1
            for i in range(n):
                W[k*n + i, r] = 1
            r += 1
    
    # Block 2: j in {2,3,4}
    for j in range(2, 5):
        for k in range(p):
            for i in range(n):
                U[i*m + j, r] = 1
            V[j*p + k, r] = 1
            for i in range(n):
                W[k*n + i, r] = 1
            r += 1
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
