import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    The known minimal rank for this tensor is 11.
    """
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize matrices with zeros
    # U: (n*m, rank) -> 4x11
    # V: (m*p, rank) -> 6x11
    # W: (p*n, rank) -> 6x11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper functions to set values based on logical indices
    # U[i, j, r] corresponds to U[i*m + j][r]
    # V[j, k, r] corresponds to V[j*p + k][r]
    # W[k, i, r] corresponds to W[k*n + i][r]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
        
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
        
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # The decomposition below is a known optimal rank 11 decomposition for 2x2x3.
    # It combines several bilinear forms.
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # Contributes to C00, C11
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1) # C00 += 1
    set_w(1, 1, 0, 1) # C11 += 1

    # Term 2: (A10 + A11) * B00
    # Contributes to C10, C11
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1) # C10 += 1
    set_w(1, 1, 1, -1) # C11 -= 1

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1) # C01 += 1
    set_w(1, 1, 2, 1) # C11 += 1

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1) # C00 += 1
    set_w(0, 1, 3, 1) # C10 += 1

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1) # C00 -= 1
    set_w(1, 0, 4, 1) # C01 += 1

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1) # C11 += 1

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1) # C00 += 1

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1) # C02 += 1

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1) # C02 += 1

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1) # C12 += 1

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1) # C12 += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
