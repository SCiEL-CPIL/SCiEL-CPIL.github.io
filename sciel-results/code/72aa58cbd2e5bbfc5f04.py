"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose the
    matrix multiplication tensor. Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition with rank n*m*p = 54
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify correctness: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError("Decomposition verification failed")
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        raise TimeoutError(f"Exceeded time limit: {elapsed:.1f}s > {deadline:.1f}s")
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
