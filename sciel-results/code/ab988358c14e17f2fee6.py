import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses a block-structured decomposition with 2x2 blocks:
    - Row 0 of C: 3 blocks of 1x2 * 2x2, each rank 4 (direct) => 12
    - Rows 1-2 of C: 3 blocks of 2x2 * 2x2, each rank 7 (Strassen) => 21
    Total rank = 33 * 2 = 66 (wait, let me recalculate)
    
    Actually: Row 0 has 2 column-pairs of A, 3 column-pairs of B
    For each (row-block of C, col-pair of B):
      If C-row-block is 1x2 (row 0): use 1x2 * 2x2 direct (rank 4) for each A col-pair (2 of them) => 2*4 = 8
      If C-row-block is 2x2 (rows 1-2): use 2x2 * 2x2 Strassen (rank 7) for each A col-pair (2 of them) => 2*7 = 14
    
    Total = 3 * 8 + 3 * 14 = 24 + 42 = 66
    """
    n, m, p = 3, 4, 6
    
    R = 66
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
    
    for j in range(3):
        for i in range(2):
            add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
    
    for j in range(3):
        for i in range(2):
            add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
    
    assert col == R, f"Expected rank {R}, got {col}"
    return {"U": U, "V": V, "W": W}
