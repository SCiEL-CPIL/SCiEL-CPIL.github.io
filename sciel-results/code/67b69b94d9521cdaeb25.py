def construct():
    n, m, p = 2, 3, 10
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # U: non-zero for j == r
        for i in range(n):
            U[i * m + r][r] = 1
        # V: non-zero for j == r
        for k in range(p):
            V[r * p + k][r] = 1
        # W: non-zero for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
