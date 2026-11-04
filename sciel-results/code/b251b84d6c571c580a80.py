"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Uses the schoolbook decomposition which has rank n*m*p = 75.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill the matrices for the schoolbook decomposition
    # The tensor is decomposed as sum_{i,j,k} e_{i,j}^U \otimes e_{j,k}^V \otimes e_{i,k}^W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (following the trace convention W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    # Convert to lists if required by the interface, though numpy arrays are usually accepted
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is standard.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
