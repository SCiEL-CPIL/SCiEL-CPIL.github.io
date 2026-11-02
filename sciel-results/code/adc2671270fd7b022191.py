"""Optimized program using Lanczos reduction for 3x4x4 matrix multiplication."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-48 exact tensor decomposition for multiplying 
    a 3x4 matrix by a 4x4 matrix using the schoolbook method.
    Although rank 35 is theoretically possible, this implementation
    guarantees validity and correctness using standard linear algebra.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use numpy for efficient array initialization
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Flatten indices according to the trace convention
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
