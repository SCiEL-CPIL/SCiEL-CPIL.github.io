def construct():
    n, m, p = 3, 5, 5
    rank = 3
    U = [[0] * rank for _ in range(n * m)]  # (15, 3)
    V = [[0] * rank for _ in range(m * p)]  # (25, 3)
    W = [[0] * rank for _ in range(p * n)]  # (15, 3)
    
    for i in range(n):
        r = i
        # U: U[i*m + j, r] = 1 for all j
        for j in range(m):
            U[i * m + j][r] = 1
        # V: V[j*p + k, r] = 1 if k == i, else 0
        for j in range(m):
            V[j * p + i][r] = 1
        # W: W[k*n + i, r] = 1 if k == i, else 0
        W[i * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
