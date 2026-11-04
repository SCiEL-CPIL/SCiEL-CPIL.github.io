"""
Program for Matrix Multiplication 3x3x3 Tensor Decomposition.
Returns the schoolbook decomposition of rank 27.
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-27 schoolbook decomposition for 3x3 matrix multiplication.
    U: (9, 27)
    V: (9, 27)
    W: (9, 27)
    Entries are 0 or 1 (integers).
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    break
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i, col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke early, the matrices are partially filled, which is invalid.
    # However, the loop is so fast it will never break.
    # We return the full rank 27 decomposition.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
