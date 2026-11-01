"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_rank7():
    """
    Returns the known valid rank-7 Strassen decomposition for 2x2 matrix multiply.
    This is the minimal possible rank for 2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ]
    
    # W indices: k*n + i
    # Row 0: k=0, i=0 -> C00
    # Row 1: k=0, i=1 -> C10
    # Row 2: k=1, i=0 -> C01
    # Row 3: k=1, i=1 -> C11
    
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 coeffs: M1, M4, -M5, M7
        [0, 1, 0, 1, 0, 0, 0],   # C10 coeffs: M2, M4
        [0, 0, 1, 0, 1, 0, 0],   # C01 coeffs: M3, M5
        [1, -1, 1, 0, 0, 1, 0],  # C11 coeffs: M1, -M2, M3, M6
    ]
    
    return (np.array(U_data, dtype=float),
            np.array(V_data, dtype=float),
            np.array(W_data, dtype=float))

def construct():
    """
    Returns the rank-minimal (rank 7) exact tensor decomposition for 2x2 matrix multiplication.
    Since 7 is the proven lower bound for the tensor rank of 2x2 matrix multiplication,
    this is the optimal solution.
    """
    U, V, W = _strassen_rank7()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
