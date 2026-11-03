DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 3 * 9 = 27
    # U: (n*m, rank) = (6, 27)
    # V: (m*p, rank) = (27, 27)
    # W: (p*n, rank) = (18, 27)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
