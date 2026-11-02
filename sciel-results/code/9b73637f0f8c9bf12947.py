"""Program for exact tensor decomposition of 2x4 * 4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # Approach: Direct sum of row-wise dot products.
    # The tensor T decomposes into n independent tensors T_i, one for each row i of C.
    # T_i computes C[i, :] = A[i, :] * B.
    # The rank of T_i (1xm * mxp) is m*p.
    # Total rank is sum_{i=0}^{n-1} m*p = n*m*p.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Iterate over each row i of A (and C)
    for i in range(n):
        # Iterate over each column k of B (and C)
        # The term for C[i, k] is sum_j A[i, j] * B[j, k]
        # This requires m multiplications (one for each j)
        for k in range(p):
            for j in range(m):
                # U index: i * m + j
                U[i * m + j, col] = 1
                # V index: j * p + k
                V[j * p + k, col] = 1
                # W index: k * n + i (trace convention W[k*n+i])
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists to match the interface expected by the evaluator
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
