"""Schoolbook decomposition for 2x2 * 2x15 matrix multiplication.

New approach: direct schoolbook (naive) decomposition.
Each output element C[i,k] = sum_j A[i,j] * B[j,k]
Total rank = n * m * p = 2 * 2 * 15 = 60.
"""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    rank = n * m * p  # 60
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 60
    V = [[0] * rank for _ in range(m * p)]  # 30 x 60
    W = [[0] * rank for _ in range(p * n)]  # 30 x 60
    
    r = 0
    for i in range(n):       # row of A, row of C
        for j in range(m):   # col of A, row of B
            for k in range(p):  # col of B, col of C
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
