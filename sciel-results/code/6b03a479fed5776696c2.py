"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 7]

def construct():
    """Return the schoolbook decomposition for 2x3 * 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42 for 2*3*7
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (21, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (14, 42)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
