def construct():
    n, m, p = 2, 4, 4
    rank = m  # 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V based on j (the shared index)
    # U index: i*m + j. Col: c.
    # V index: j*p + k. Col: c.
    # W index: k*n + i. Col: c.
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
            
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
            
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
