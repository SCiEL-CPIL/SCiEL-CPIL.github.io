"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    The coefficients are integers, ensuring exactness.
    """
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # Row order: A[0,0], A[0,1], A[1,0], A[1,1]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # Row order: B[0,0], B[0,1], B[1,0], B[1,1]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # Row index = k*n + i (official trace convention)
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]
    # Row 2: k=1, i=0 -> C[0,1]
    # Row 3: k=1, i=1 -> C[1,1]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    return {"U": U_data, "V": V_data, "W": W_data}
# EVOLVE-BLOCK-END
