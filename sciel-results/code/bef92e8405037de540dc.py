<<<<<<< SEARCH
"""Program for 6x6x6 matrix multiplication tensor decomposition using CW(3x3) ⊗ Strassen(2x2)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen decomposition for 2x2 matrices, rank 7
    # U2 shape: (4, 7)  -- indices (i,j) where i in 0..1, j in 0..1
    # V2 shape: (4, 7)  -- indices (j,k) where j in 0..1, k in 0..1
    # W2 shape: (4, 7)  -- indices (k,i) where k in 0..1, i in 0..1
    #
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    U2 = [
        [1, 1, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 1, 0, -1, 1],  # A10
        [0, 0, 0, 0, 1, 0, 0],   # A01
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 0],   # B01
        [1, 0, -1, -1, 1, 0, 1], # B11
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, -1], # C00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1],  # C11 = M1 - M2 + M3 + M7
    ]
=======
"""Program for 6x6x6 matrix multiplication tensor decomposition using CW(3x3) ⊗ Strassen(2x2)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen decomposition for 2x2 matrices, rank 7
    U2 = [
        [1, 1, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 1, 0, -1, 1],  # A10
        [0, 0, 0, 0, 1, 0, 0],   # A01
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 0],   # B01
        [1, 0, -1, -1, 1, 0, 1], # B11
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, -1], # C00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1],  # C11 = M1 - M2 + M3 + M7
    ]
>>>>>>> REPLACE

Now I need to add the rank-23 decomposition for 3x3 and the Kronecker product construction. Let me replace the rest of the function:

