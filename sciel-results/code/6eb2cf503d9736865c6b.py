"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Use Winograd Fourier Transform Algorithm (WFTA) approach
    # Decompose 2x3x10 multiplication using Kronecker product structure
    # 
    # Strategy: Pad dimensions to powers of 2 and use WFTA
    # 3 -> 4, 10 -> 16
    # But we can be smarter: use the fact that we can decompose into smaller blocks
    #
    # For now, let's try a direct approach using the structure:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can use the fact that this is equivalent to multiple 1x1x1 multiplications
    # but we want to reduce the count.
    #
    # Let's try using the Winograd algorithm for small matrix multiplication
    # The key is to find a decomposition that uses fewer than n*m*p terms
    #
    # For 2x3x10, let's try to use the structure:
    # We can write this as:
    # C[0,0:10] = A[0,0]*B[0,0:10] + A[0,1]*B[1,0:10] + A[0,2]*B[2,0:10]
    # C[1,0:10] = A[1,0]*B[0,0:10] + A[1,1]*B[1,0:10] + A[1,2]*B[2,0:10]
    #
    # This is already the schoolbook with 60 terms.
    #
    # To reduce, we need to find a tensor decomposition with rank < 60.
    # 
    # Let's try a different approach: use the fact that we can combine some terms
    # using the linearity of matrix multiplication.
    #
    # Actually, let's just return the schoolbook decomposition for now, 
    # but structured properly. The minimum rank for 2x3x10 is known to be less than 60.
    #
    # For safety, return the schoolbook method which is guaranteed correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
