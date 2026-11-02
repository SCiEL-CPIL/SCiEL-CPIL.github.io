"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Schoolbook decomposition
    # Rank = n * p = 2 * 13 = 26
    # Each term corresponds to A[i,j] * B[j,k] contributing to C[i,k]
    
    rank = n * p  # 26
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 26
    V = [[0] * rank for _ in range(m * p)] # 26 x 26
    W = [[0] * rank for _ in range(p * n)] # 26 x 26
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][col] = 1
                # V index: j * p + k
                V[j * p + k][col] = 1
                # W index: k * n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
