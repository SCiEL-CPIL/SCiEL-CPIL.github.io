"""Matrix multiplication 3x4x4 rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We use the trivial decomposition which gives rank = n*m*p = 48
    # But let's try to find a better one using the structure of the problem.
    
    # For 3x4 x 4x4 matrix multiplication:
    # Output C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in [0,3), j in [0,4), k in [0,4)
    
    # The tensor has indices (i,j,k) with values 1 if we're computing C[i,k]
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (16, rank)  
    # W has shape (p*n, rank) = (12, rank)
    
    # Index conventions:
    # U[i*m + j, r] - coefficient for A[i,j] in rank-r term
    # V[j*p + k, r] - coefficient for B[j,k] in rank-r term
    # W[k*n + i, r] - coefficient for C[i,k] in rank-r term (trace convention)
    
    # Trivial decomposition: rank = 48
    rank = n * m * p
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
# EVOLVE-BLOCK-END
