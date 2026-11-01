"""Rewritten program for matrix multiplication 2,2,9 tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Schoolbook rank = n*m*p = 36
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]  # shape (4, rank)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (18, rank)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (18, rank)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
