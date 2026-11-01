"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (provably optimal rank for 2x2).
    Rank 6 is impossible: the tensor rank of 2x2 matrix multiplication is 7.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2

    # Strassen's 7 products coefficients
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]

    V_data = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]

    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    return {"U": U_data, "V": V_data, "W": W_data}
# EVOLVE-BLOCK-END
