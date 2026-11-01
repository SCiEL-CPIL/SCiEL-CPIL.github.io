"""Matrix multiplication 2x2,2x7 exact tensor decomposition (Rank 25 construction)."""

import time

DIMENSIONS = [2, 2, 7]


# EVOLVE-BLOCK-START
def _add_strassen_block(U, V, W, col_start, term_offset):
    """
    Adds a Strassen block for a 2x2 submatrix of B (columns col_start, col_start+1).
    Assumes A is 2x2 with indices 0,1,2,3.
    B indices in V are j*7 + k.
    W indices are k*2 + i.
    """
    n, m, p = 2, 2, 7
    
    # B indices
    b00 = 0 * p + col_start
    b10 = 1 * p + col_start
    b01 = 0 * p + (col_start + 1)
    b11 = 1 * p + (col_start + 1)
    
    # W indices
    c00 = col_start * n + 0
    c10 = col_start * n + 1
    c01 = (col_start + 1) * n + 0
    c11 = (col_start + 1) * n + 1
    
    # Strassen Terms (M1 to M7)
    # M1 = (A00 + A11)(B00 + B11)
    U[0][term_offset + 0] += 1
    U[3][term_offset + 0] += 1
    V[b00][term_offset + 0] += 1
    V[b11][term_offset + 0] += 1
    W[c00][term_offset + 0] += 1
    W[c10][term_offset + 0] += -1
    W[c01][term_offset + 0] += 1
    W[c11][term_offset + 0] += 0
    
    # M2 = (A10 + A11)B00
    U[2][term_offset + 1] += 1
    U[3][term_offset + 1] += 1
    V[b00][term_offset + 1] += 1
    W[c00][term_offset + 1] += 0
    W[c10][term_offset + 1] += 1
    W[c01][term_offset + 1] += 0
    W[c11][term_offset + 1] += 0
    
    # M3 = A00(B10 - B11)
    U[0][term_offset + 2] += 1
    V[b10][term_offset + 2] += 1
    V[b11][term_offset + 2] += -1
    W[c00][term_offset + 2] += 0
    W[c10][term_offset + 2] += 0
    W[c01][term_offset + 2] += 1
    W[c11][term_offset + 2] += 0
    
    # M4 = A11(B01 - B00)
    U[3][term_offset + 3] += 1
    V[b01][term_offset + 3] += 1
    V[b00][term_offset + 3] += -1
    W[c00][term_offset + 3] += 1
    W[c10][term_offset + 3] += 0
    W[c01][term_offset + 3] += 0
    W[c11][term_offset + 3] += -1
    
    # M5 = (A00 + A10)B11
    U[0][term_offset + 4] += 1
    U[2][term_offset + 4] += 1
    V[b11][term_offset + 4] += 1
    W[c00][term_offset + 4] += -1
    W[c10][term_offset + 4] += 1
    W[c01][term_offset + 4] += 0
    W[c11][term_offset + 4] += 0
    
    # M6 = (A11 - A00)(B00 + B01)
    U[3][term_offset + 5] += 1
    U[0][term_offset + 5] += -1
    V[b00][term_offset + 5] += 1
    V[b01][term_offset + 5] += 1
    W[c00][term_offset + 5] += 0
    W[c10][term_offset + 5] += -1
    W[c01][term_offset + 5] += 1
    W[c11][term_offset + 5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][term_offset + 6] += 1
    U[3][term_offset + 6] += -1
    V[b10][term_offset + 6] += 1
    V[b11][term_offset + 6] += 1
    W[c00][term_offset + 6] += 1
    W[c10][term_offset + 6] += 0
    W[c01][term_offset + 6] += -1
    W[c11][term_offset + 6] += 0
    
    # Correction: The W contributions above are the coefficients for the reconstruction.
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M5 - M1 - M6
    # C01 = M3 + M6 + M1 - M7
    # C11 = M3 + M2 - M4 + M6
    
    # Let's re-apply the coefficients correctly to W.
    # I initialized W with 0. I need to set the specific coefficients.
    
    # Reset W for this block's terms to 0 to be safe, then apply.
    for k in range(7):
        W[c00][term_offset + k] = 0
        W[c10][term_offset + k] = 0
        W[c01][term_offset + k] = 0
        W[c11][term_offset + k] = 0

    # M1
    W[c00][term_offset + 0] += 1
    W[c10][term_offset + 0] += -1
    W[c01][term_offset + 0] += 1
    W[c11][term_offset + 0] += 0
    
    # M2
    W[c00][term_offset + 1] += 0
    W[c10][term_offset + 1] += 1
    W[c01][term_offset + 1] += 0
    W[c11][term_offset + 1] += 0
    
    # M3
    W[c00][term_offset + 2] += 0
    W[c10][term_offset + 2] += 0
    W[c01][term_offset + 2] += 1
    W[c11][term_offset + 2] += 0
    
    # M4
    W[c00][term_offset + 3] += 1
    W[c10][term_offset + 3] += 0
    W[c01][term_offset + 3] += 0
    W[c11][term_offset + 3] += -1
    
    # M5
    W[c00][term_offset + 4] += -1
    W[c10][term_offset + 4] += 1
    W[c01][term_offset + 4] += 0
    W[c11][term_offset + 4] += 0
    
    # M6
    W[c00][term_offset + 5] += 0
    W[c10][term_offset + 5] += -1
    W[c01][term_offset + 5] += 1
    W[c11][term_offset + 5] += 1
    
    # M7
    W[c00][term_offset + 6] += 1
    W[c10][term_offset + 6] += 0
    W[c01][term_offset + 6] += -1
    W[c11][term_offset + 6] += 0


def _add_standard_block(U, V, W, col, term_offset):
    """
    Adds a standard 2x1 multiplication block for column 'col' of B.
    """
    n, m, p = 2, 2, 7
    
    # B indices
    b0 = 0 * p + col
    b1 = 1 * p + col
    
    # W indices
    c0 = col * n + 0
    c1 = col * n + 1
    
    # Term 0: A00 * B0 -> C0
    U[0][term_offset + 0] += 1
    V[b0][term_offset + 0] += 1
    W[c0][term_offset + 0] += 1
    
    # Term 1: A01 * B1 -> C0
    U[1][term_offset + 1] += 1
    V[b1][term_offset + 1] += 1
    W[c0][term_offset + 1] += 1
    
    # Term 2: A10 * B0 -> C1
    U[2][term_offset + 2] += 1
    V[b0][term_offset + 2] += 1
    W[c1][term_offset + 2] += 1
    
    # Term 3: A11 * B1 -> C1
    U[3][term_offset + 3] += 1
    V[b1][term_offset + 3] += 1
    W[c1][term_offset + 3] += 1


def construct():
    n, m, p = DIMENSIONS
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Cols 0, 1 (Strassen)
    _add_strassen_block(U, V, W, 0, 0)
    
    # Block 2: Cols 2, 3 (Strassen)
    _add_strassen_block(U, V, W, 2, 7)
    
    # Block 3: Cols 4, 5 (Strassen)
    _add_strassen_block(U, V, W, 4, 14)
    
    # Block 4: Col 6 (Standard)
    _add_standard_block(U, V, W, 6, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
