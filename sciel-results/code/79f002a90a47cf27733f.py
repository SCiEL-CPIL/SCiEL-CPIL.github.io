"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    # 3 * 7 (Strassen 2x2) + 3 * 7 (Strassen 2x2) + 12 (Schoolbook 2x1x6)
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: i (0,1), j (0,1) -> idx = i*2 + j
    # Indices for B: j (0,1), k (0,1) -> idx = j*2 + k
    # Indices for C: k (0,1), i (0,1) -> idx = k*2 + i (Trace convention W[k*n+i, r])
    # Note: W index is k*n + i. Here n=2 for the sub-block.
    # But in the full matrix, n=2, p=6.
    # The sub-block C is at specific i, k positions.
    
    def add_strassen(a_row_offset, b_row_offset, c_row_offset, c_col_offset, start_col):
        # a_row_offset: row index in A for the 2x2 block (0 or 2)
        # b_row_offset: row index in B for the 2x2 block (0 or 2)
        # c_row_offset: row index in C for the 2x2 block (0, 2, or 4)
        # c_col_offset: col index in C for the 2x2 block (0 or 1)
        
        # Strassen M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 terms.
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: A00, A11
            # V: B00, B11
            # W: C00, C11
            (([0, 0, 0, 1], [1, 0, 0, 1]), ([0, 1], [1, 0]), ([1, 0, 0, 1], [0, 1, 0, 0]) ), # M1? No, this is messy.
        ]
        
        # Let's just hardcode the 7 terms for 2x2x2 Strassen.
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 10, 11
        # C indices: 00, 01, 10, 11 (k*n+i)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[a_row_offset*2 + 0][start_col] = 1
        U[a_row_offset*2 + 1][start_col] = 1
        V[b_row_offset*2 + 0][start_col] = 1
        V[b_row_offset*2 + 1][start_col] = 1
        W[c_row_offset*n + c_col_offset*2 + 0][start_col] = 1 # C00
        W[c_row_offset*n + c_col_offset*2 + 1][start_col] = 1 # C11? No.
        # C indices: k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
        # C00 (k=0, i=0) -> idx 0.
        # C11 (k=1, i=1) -> idx 3.
        W[c_row_offset*n + 0][start_col] = 1
        W[c_row_offset*n + 3][start_col] = 1
        start_col += 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=1 (with -1 for C11? No, M2 is added to C10 and subtracted from C11)
        # C10 (k=1, i=0) -> idx 2.
        # C11 (k=1, i=1) -> idx 3.
        U[a_row_offset*2 + 2][start_col] = 1
        U[a_row_offset*2 + 3][start_col] = 1
        V[b_row_offset*2 + 0][start_col] = 1
        W[c_row_offset*n + 2][start_col] = 1
        W[c_row_offset*n + 3][start_col] = -1
        start_col += 1
        
        # Term 3: M3 = A00(B10-B11)
        # U: A00=1
        # V: B10=1, B11=-1
        # W: C01=1, C11=1
        # C01 (k=0, i=1) -> idx 1.
        # C11 (k=1, i=1) -> idx 3.
        U[a_row_offset*2 + 0][start_col] = 1
        V[b_row_offset*2 + 2][start_col] = 1
        V[b_row_offset*2 + 3][start_col] = -1
        W[c_row_offset*n + 1][start_col] = 1
        W[c_row_offset*n + 3][start_col] = 1
        start_col += 1
        
        # Term 4: M4 = A11(B01-B00)
        # U: A11=1
        # V: B01=1, B00=-1
        # W: C00=1, C10=1
        # C00 (k=0, i=0) -> idx 0.
        # C10 (k=1, i=0) -> idx 2.
        U[a_row_offset*2 + 3][start_col] = 1
        V[b_row_offset*2 + 1][start_col] = 1
        V[b_row_offset*2 + 0][start_col] = -1
        W[c_row_offset*n + 0][start_col] = 1
        W[c_row_offset*n + 2][start_col] = 1
        start_col += 1
        
        # Term 5: M5 = (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        # C00 (k=0, i=0) -> idx 0.
        # C01 (k=0, i=1) -> idx 1.
        U[a_row_offset*2 + 0][start_col] = 1
        U[a_row_offset*2 + 1][start_col] = 1
        V[b_row_offset*2 + 3][start_col] = 1
        W[c_row_offset*n + 0][start_col] = -1
        W[c_row_offset*n + 1][start_col] = 1
        start_col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        # U: A10=1, A00=-1
        # V: B00=1, B10=1
        # W: C11=1
        # C11 (k=1, i=1) -> idx 3.
        U[a_row_offset*2 + 2][start_col] = 1
        U[a_row_offset*2 + 0][start_col] = -1
        V[b_row_offset*2 + 0][start_col] = 1
        V[b_row_offset*2 + 2][start_col] = 1
        W[c_row_offset*n + 3][start_col] = 1
        start_col += 1
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        # U: A01=1, A11=-1
        # V: B01=1, B11=1
        # W: C00=1
        # C00 (k=0, i=0) -> idx 0.
        U[a_row_offset*2 + 1][start_col] = 1
        U[a_row_offset*2 + 3][start_col] = -1
        V[b_row_offset*2 + 1][start_col] = 1
        V[b_row_offset*2 + 3][start_col] = 1
        W[c_row_offset*n + 0][start_col] = 1
        start_col += 1
        
        return start_col

    # Block 1: A[0:2, 0:2] * B[0:2, 0:6]
    # B is split into 3 blocks of 2 cols: 0:2, 2:4, 4:6
    col = add_strassen(0, 0, 0, 0, col) # C[0:2, 0:2]
    col = add_strassen(0, 0, 0, 1, col) # C[0:2, 2:4]
    col = add_strassen(0, 0, 0, 2, col) # C[0:2, 4:6]
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:6]
    col = add_strassen(2, 2, 0, 0, col) # C[0:2, 0:2]
    col = add_strassen(2, 2, 0, 1, col) # C[0:2, 2:4]
    col = add_strassen(2, 2, 0, 2, col) # C[0:2, 4:6]
    
    # Block 3: A[0:2, 4:5] * B[4:5, 0:6]
    # Schoolbook for 2x1x6
    # A[0,4], A[1,4]
    # B[4,0]...B[4,5]
    # C[0,0]...C[1,5]
    for k in range(6):
        # Term for C[0, k]
        U[4][col] = 1
        V[4*6 + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        # Term for C[1, k]
        U[5][col] = 1
        V[4*6 + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
