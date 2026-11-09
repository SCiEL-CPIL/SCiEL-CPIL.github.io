def construct():
    n, m, p = DIMENSIONS
    
    # General case: rank = m
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    for l in range(m):
        for k in range(p):
            V[l * p + k][l] = 1
    
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
