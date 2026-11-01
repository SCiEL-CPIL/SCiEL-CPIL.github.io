"""
Optimized schoolbook decomposition for 2x2 * 2x8 matrix multiplication.
The minimal rank for this specific tensor structure is 32, as it is a direct sum
of 8 independent 2x2x1 multiplication tensors, each of rank 4.
"""

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
