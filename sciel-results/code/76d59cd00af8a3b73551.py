"""Tensor decomposition for 3x3x8 matrix multiplication."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    rank = 19

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # For each of the 3 output columns, embed a 3x3x3 rank-19 decomposition.
    # Indices: U row = i*m + j, V row = j*p + k, W row = k*n + i
    # For column c = k_offset (0..2), we use a 3x3x3 rank-19 scheme.
    # We'll use a known rank-19 decomposition for 3x3x3.
    
    # Strassen-like or known rank-19 for 3x3x3:
    # I'll use a simple construction: for each k in 0..2, use 19 terms from a 3x3x3 rank-19.
    # Since I don't have a verified rank-19 for 3x3x3, I'll fall back to schoolbook 
    # but try to find a better one.
    
    # Actually, let me just use schoolbook rank 72 to be safe, since I cannot verify
    # a rank-19 decomposition for 3x3x3 off the top of my head.
    rank = n * m * p  # 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
