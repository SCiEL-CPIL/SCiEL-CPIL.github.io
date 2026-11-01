DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # 3
    # U: (n*m) x rank = 6 x 3
    # U[i*m + j, r] = 1 if j == r else 0
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if j == r:
                    U[i * m + j][r] = 1
    
    # V: (m*p) x rank = 27 x 3
    # V[j'*p + k, r] = 1 if j' == r else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if j == r:
                    V[j * p + k][r] = 1
    
    # W: (p*n) x rank = 18 x 3
    # W[k*n + i, r] = 1 for all k, i, r
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
