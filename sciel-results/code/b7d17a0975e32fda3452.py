DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n  # 2 terms, one per row of A
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        c = i  # column index in U, V, W
        # U[i*m+j, c] = 1 for all j
        for j in range(m):
            U[i * m + j][c] = 1
        # V[j*p+k, c] = 1 for all j, k
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = 1
        # W[k*n+i, c] = 1 for all k
        for k in range(p):
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
