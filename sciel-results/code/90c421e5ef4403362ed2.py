DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition for 3x3x6 matrix multiplication.
    # This is equivalent to two independent 3x3x3 multiplications.
    # Rank of 3x3x3 schoolbook is 27.
    # Total rank is 27 + 27 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
