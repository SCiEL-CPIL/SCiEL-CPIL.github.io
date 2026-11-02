def _build_rank12():
    n, m, p = 2, 2, 6
    rank = 12
    U = [[0]*rank for _ in range(n*m)]  # (4, 12)
    V = [[0]*rank for _ in range(m*p)]  # (12, 12)
    W = [[0]*rank for _ in range(p*n)]  # (12, 12)
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            j = k % m  # j = k % 2
            
            U[i*m + j][r] = 1
            V[j*p + k][r] = 1
            W[k*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
