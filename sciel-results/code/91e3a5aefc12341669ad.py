"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 4x4 * 4x9 matrix multiplication:
    # The tensor T has entries T[i*m+k, k*p+j, j*n+i] = 1 for all valid (i,k,j)
    # and 0 otherwise.
    # 
    # The schoolbook rank is n*m*p = 4*4*9 = 144.
    # 
    # For this specific case, we can use a structured decomposition.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # One approach: decompose using the fact that for each k, the contribution
    # to C[:,j] from A[:,k] * B[k,j] is an outer product.
    # 
    # Rank = n * m * p = 144 (schoolbook)
    # 
    # Can we do better? For matrix multiplication, the lower bound is max(n*m, m*p, n*p) = max(16, 36, 36) = 36.
    # 
    # Let's try to find a decomposition with rank < 144.
    # 
    # Approach: Use the standard schoolbook but see if we can share computations.
    # Actually, for exact tensor decomposition of matrix multiplication, the rank
    # equals the number of scalar multiplications needed.
    # 
    # For 4x4 * 4x9, we can use Strassen-like methods on the 4x4 part.
    # But the 4x9 part complicates things.
    # 
    # Let's just use schoolbook rank 144 for now, which is guaranteed to work.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
