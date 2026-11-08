def construct():
    n, m, p = 3, 3, 6
    rank = n * m  # 9
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U index: i*m + j
            U[i * m + j][col] = 1
            # V indices: j*p + k for k in 0..p-1
            for k in range(p):
                V[j * p + k][col] = 1
            # W indices: k*n + i for k in 0..p-1
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
