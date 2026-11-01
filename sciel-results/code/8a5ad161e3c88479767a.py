DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = m * p  # 16
    U = [[0] * rank for _ in range(n * m)]  # 8 x 16
    V = [[0] * rank for _ in range(m * p)]  # 16 x 16
    W = [[0] * rank for _ in range(p * n)]  # 8 x 16
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i*m + j, c] = 1 for i in range(n)
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for i in range(n)
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
