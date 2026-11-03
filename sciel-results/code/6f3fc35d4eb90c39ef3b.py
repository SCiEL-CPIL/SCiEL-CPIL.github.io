def construct():
    n, m, p = 3, 5, 6
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        j = c // p
        k = c % p
        for i in range(n):
            U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        for i in range(n):
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
