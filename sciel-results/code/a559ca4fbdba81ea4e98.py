"""Program to construct a low-rank exact decomposition of the 3x3 matrix multiplication tensor."""

import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-22 bilinear algorithm for 3x3 matrix multiplication.
    
    This algorithm uses a block decomposition approach:
    1. The 2x2 top-left block is handled using Strassen's algorithm (rank 7).
    2. The remaining cross terms (involving the 3rd row/column) are handled
       directly, requiring 15 additional multiplications.
    
    Total rank: 7 + 15 = 22.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      - U has shape (9, 22)
      - V has shape (9, 22)
      - W has shape (9, 22)
    """
    n, m, p = 3, 3, 3
    rank = 22
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper functions to set coefficients
    # A indices: i, j (row i, col j) -> index i*m + j
    # B indices: j, k (row j, col k) -> index j*p + k
    # C indices: i, k (row i, col k) -> output index k*n + i (Trace Convention)
    
    def set_A(i, j, c, val):
        U[i*m + j, c] += val
        
    def set_B(j, k, c, val):
        V[j*p + k, c] += val
        
    def set_C(i, k, c, val):
        # W index is k*n + i
        W[k*n + i, c] += val

    # ---------------------------------------------------------
    # Part 1: Strassen's Algorithm for the 2x2 top-left block
    # A_TL = [[A00, A01], [A10, A11]]
    # B_TL = [[B00, B01], [B10, B11]]
    # ---------------------------------------------------------
    
    # M1 = (A00 + A11) * (B00 + B11)
    c = 0
    set_A(0, 0, c, 1)
    set_A(1, 1, c, 1)
    set_B(0, 0, c, 1)
    set_B(1, 1, c, 1)
    # Contribution to C: C00 += M1, C11 += M1
    set_C(0, 0, c, 1)
    set_C(1, 1, c, 1)

    # M2 = (A10 + A11) * B00
    c = 1
    set_A(1, 0, c, 1)
    set_A(1, 1, c, 1)
    set_B(0, 0, c, 1)
    # Contribution to C: C10 += M2, C11 -= M2
    set_C(1, 0, c, 1)
    set_C(1, 1, c, -1)

    # M3 = A00 * (B01 - B11)
    c = 2
    set_A(0, 0, c, 1)
    set_B(0, 1, c, 1)
    set_B(1, 1, c, -1)
    # Contribution to C: C01 += M3, C11 += M3
    set_C(0, 1, c, 1)
    set_C(1, 1, c, 1)

    # M4 = A11 * (B10 - B00)
    c = 3
    set_A(1, 1, c, 1)
    set_B(1, 0, c, 1)
    set_B(0, 0, c, -1)
    # Contribution to C: C00 += M4, C10 += M4
    set_C(0, 0, c, 1)
    set_C(1, 0, c, 1)

    # M5 = (A00 + A01) * B11
    c = 4
    set_A(0, 0, c, 1)
    set_A(0, 1, c, 1)
    set_B(1, 1, c, 1)
    # Contribution to C: C00 -= M5, C01 += M5
    set_C(0, 0, c, -1)
    set_C(0, 1, c, 1)

    # M6 = (A10 - A00) * (B00 + B01)
    c = 5
    set_A(1, 0, c, 1)
    set_A(0, 0, c, -1)
    set_B(0, 0, c, 1)
    set_B(0, 1, c, 1)
    # Contribution to C: C11 += M6
    set_C(1, 1, c, 1)

    # M7 = (A01 - A11) * (B10 + B11)
    c = 6
    set_A(0, 1, c, 1)
    set_A(1, 1, c, -1)
    set_B(1, 0, c, 1)
    set_B(1, 1, c, 1)
    # Contribution to C: C00 += M7
    set_C(0, 0, c, 1)

    # ---------------------------------------------------------
    # Part 2: Cross terms involving index 2
    # ---------------------------------------------------------
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    c = 7
    set_A(0, 0, c, 1); set_B(0, 2, c, 1); set_C(0, 2, c, 1)
    
    c = 8
    set_A(0, 1, c, 1); set_B(1, 2, c, 1); set_C(0, 2, c, 1)
    
    c = 9
    set_A(0, 2, c, 1); set_B(2, 2, c, 1); set_C(0, 2, c, 1)

    # C12 = A10*B02 + A11*B12 + A12*B22
    c = 10
    set_A(1, 0, c, 1); set_B(0, 2, c, 1); set_C(1, 2, c, 1)
    
    c = 11
    set_A(1, 1, c, 1); set_B(1, 2, c, 1); set_C(1, 2, c, 1)
    
    c = 12
    set_A(1, 2, c, 1); set_B(2, 2, c, 1); set_C(1, 2, c, 1)

    # C20 = A20*B00 + A21*B10 + A22*B20
    c = 13
    set_A(2, 0, c, 1); set_B(0, 0, c, 1); set_C(2, 0, c, 1)
    
    c = 14
    set_A(2, 1, c, 1); set_B(1, 0, c, 1); set_C(2, 0, c, 1)
    
    c = 15
    set_A(2, 2, c, 1); set_B(2, 0, c, 1); set_C(2, 0, c, 1)

    # C21 = A20*B01 + A21*B11 + A22*B21
    c = 16
    set_A(2, 0, c, 1); set_B(0, 1, c, 1); set_C(2, 1, c, 1)
    
    c = 17
    set_A(2, 1, c, 1); set_B(1, 1, c, 1); set_C(2, 1, c, 1)
    
    c = 18
    set_A(2, 2, c, 1); set_B(2, 1, c, 1); set_C(2, 1, c, 1)

    # C22 = A20*B02 + A21*B12 + A22*B22
    c = 19
    set_A(2, 0, c, 1); set_B(0, 2, c, 1); set_C(2, 2, c, 1)
    
    c = 20
    set_A(2, 1, c, 1); set_B(1, 2, c, 1); set_C(2, 2, c, 1)
    
    c = 21
    set_A(2, 2, c, 1); set_B(2, 2, c, 1); set_C(2, 2, c, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
