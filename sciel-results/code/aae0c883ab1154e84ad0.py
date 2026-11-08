def construct():
    n, m, p = 4, 4, 8
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: k in 0..3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                r = i * 16 + j * 4 + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    # Part 2: k in 4..7
    for i in range(n):
        for j in range(m):
            for k in range(4):
                r = 16 + i * 16 + j * 4 + k
                U[i * m + j][r] = 1
                V[j * p + (k + 4)][r] = 1
                W[(k + 4) * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
