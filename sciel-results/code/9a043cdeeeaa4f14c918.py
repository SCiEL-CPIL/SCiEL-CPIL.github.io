DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    rank = m * p  # 25
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 25
    V = [[0] * rank for _ in range(m * p)]  # 25 x 25
    W = [[0] * rank for _ in range(p * n)]  # 10 x 25
    
    for j in range(m):
        for k in range(p):
            c = j * p + k  # column index in rank
            # U: nonzero at rows i*m + j for i in {0, 1}
            U[0 * m + j][c] = 1
            U[1 * m + j][c] = 1
            # V: nonzero at row j*p + k
            V[j * p + k][c] = 1
            # W: nonzero at rows k*n + i for i in {0, 1}
            W[k * n + 0][c] = 1
            W[k * n + 1][c] = 1
    
    return {"U": U, "V": V, "W": W}
