def construct():
    n, m, p = 2, 3, 9
    rank = m * p  # 27
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: both rows of A for column j
            U[0 * m + j][col] = 1.0  # i=0, j
            U[1 * m + j][col] = 1.0  # i=1, j
            # V: row j, column k of B
            V[j * p + k][col] = 1.0
            # W: column k, both rows of C
            W[k * n + 0][col] = 1.0
            W[k * n + 1][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
