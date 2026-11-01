"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    This is Strassen's algorithm, which is provably optimal for 2x2 matrices.
    Rank 7 is the theoretical minimum.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2

    # Strassen's 7 products
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    
    # Indices:
    # U row index: i*m + j (i=0..1, j=0..1)
    # V row index: j*p + k (j=0..1, k=0..1)
    # W row index: k*n + i (k=0..1, i=0..1) -> Note: W[k*n+i, r]
    
    # Standard Strassen formulation:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U coefficients for A (rows: A00, A01, A10, A11)
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A01=1
    # M6: A10=1, A00=-1
    # M7: A01=1, A11=-1
    
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]

    # V coefficients for B (rows: B00, B01, B10, B11)
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B10=1, B11=-1
    # M4: B01=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B01=1
    # M7: B10=1, B11=1
    
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 1, 0],   # B01
        [0, 0, 1, 0, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]

    # W coefficients for C (rows: C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_data = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]

    return {"U": U_data, "V": V_data, "W": W_data}
# EVOLVE-BLOCK-END
