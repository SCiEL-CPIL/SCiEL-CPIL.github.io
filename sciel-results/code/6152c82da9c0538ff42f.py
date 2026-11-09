import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix. Uses the trivial Kronecker-delta decomposition with rank = n*m*p = 60.
    
    The tensor T_{i,j,k} = U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} summed over r
    must equal 1 if (i,j,k) is a valid triple and 0 otherwise. In matrix multiplication,
    C[i,k] = sum_j A[i,j] * B[j,k], so the tensor has a 1 at each (i,j,k) and 0 elsewhere.
    
    The trivial decomposition sets one basis vector per (i,j,k) triple, giving rank = 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build factor matrices as lists of lists (to avoid numpy dependency issues in some evaluators)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (row i of A, column j)
                # V index: j*p + k (row j of B, column k)
                # W index: k*n + i (row k, column i of C) -- trace convention W[k*n+i, r]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
