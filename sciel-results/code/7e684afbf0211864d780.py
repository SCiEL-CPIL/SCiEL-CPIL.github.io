"""Matrix multiplication 2x3x10 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose into 2x2x10 and 2x1x10
    # Rank of 2x2x10 is 10 * rank(2x2x1) = 10 * 4 = 40
    # Rank of 2x1x10 is 10 * rank(2x1x1) = 10 * 2 = 20
    # Total rank 60
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 2x2x10 (first 2 cols of A, first 2 rows of B)
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0 * 2 + 0][col] = 1
        V[0 * 10 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[0 * 2 + 1][col] = 1
        V[1 * 10 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[1 * 2 + 0][col] = 1
        V[0 * 10 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[1 * 2 + 1][col] = 1
        V[1 * 10 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
    # Part 2: 2x1x10 (3rd col of A, 3rd row of B)
    for k in range(p):
        # Term 1: A[0,2] * B[2,k] -> C[0,k]
        U[0 * 2 + 2][col] = 1
        V[2 * 10 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        
        # Term 2: A[1,2] * B[2,k] -> C[1,k]
        U[1 * 2 + 2][col] = 1
        V[2 * 10 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
