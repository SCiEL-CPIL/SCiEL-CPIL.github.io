import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Helper to add a Strassen 2x2x2 block
    # a_off: offset for A indices (2x2)
    # b_off: offset for B indices (2x2)
    # c_off: offset for C indices (2x2)
    # col_off: offset for columns in U,V,W
    def add_strassen(a_off, b_off, c_off, col_off):
        # A indices: 0,1,2,3 relative to a_off
        # B indices: 0,1,2,3 relative to b_off
        # C indices: 0,1,2,3 relative to c_off
        # Strassen terms 1..7
        # M1 = (A11+A22)(B11+B22)
        # A11: a_off+0, A22: a_off+3
        # B11: b_off+0, B22: b_off+3
        # C11: c_off+0, C22: c_off+3
        U[a_off+0][col_off] = 1
        U[a_off+3][col_off] = 1
        V[b_off+0][col_off] = 1
        V[b_off+3][col_off] = 1
        W[c_off+0][col_off] = 1
        W[c_off+3][col_off] = 1
        
        # M2 = (A21+A22)B11
        # A21: a_off+2, A22: a_off+3
        # B11: b_off+0
        # C21: c_off+1, C22: c_off+3
        U[a_off+2][col_off+1] = 1
        U[a_off+3][col_off+1] = 1
        V[b_off+0][col_off+1] = 1
        W[c_off+1][col_off+1] = 1
        W[c_off+3][col_off+1] = -1
        
        # M3 = A11(B12-B22)
        # A11: a_off+0
        # B12: b_off+1, B22: b_off+3
        # C12: c_off+2, C22: c_off+3
        U[a_off+0][col_off+2] = 1
        V[b_off+1][col_off+2] = 1
        V[b_off+3][col_off+2] = -1
        W[c_off+2][col_off+2] = 1
        W[c_off+3][col_off+2] = 1
        
        # M4 = A22(B21-B11)
        # A22: a_off+3
        # B21: b_off+2, B11: b_off+0
        # C11: c_off+0, C21: c_off+1
        U[a_off+3][col_off+3] = 1
        V[b_off+2][col_off+3] = 1
        V[b_off+0][col_off+3] = -1
        W[c_off+0][col_off+3] = 1
        W[c_off+1][col_off+3] = 1
        
        # M5 = (A11+A12)B22
        # A11: a_off+0, A12: a_off+1
        # B22: b_off+3
        # C11: c_off+0, C12: c_off+2
        U[a_off+0][col_off+4] = 1
        U[a_off+1][col_off+4] = 1
        V[b_off+3][col_off+4] = 1
        W[c_off+0][col_off+4] = -1
        W[c_off+2][col_off+4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: a_off+2, A11: a_off+0
        # B11: b_off+0, B12: b_off+1
        # C22: c_off+3
        U[a_off+2][col_off+5] = 1
        U[a_off+0][col_off+5] = -1
        V[b_off+0][col_off+5] = 1
        V[b_off+1][col_off+5] = 1
        W[c_off+3][col_off+5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: a_off+1, A22: a_off+3
        # B21: b_off+2, B22: b_off+3
        # C11: c_off+0, C22: c_off+3
        U[a_off+1][col_off+6] = 1
        U[a_off+3][col_off+6] = -1
        V[b_off+2][col_off+6] = 1
        V[b_off+3][col_off+6] = 1
        W[c_off+0][col_off+6] = 1
        W[c_off+3][col_off+6] = 1

    # Part 1: A1 (2x2) * B1 (2x8)
    # A1 is rows 0-1 of A (indices 0-3)
    # B1 is rows 0-1 of B (indices 0-15)
    # C1 is rows 0-1 of C (indices 0-7)
    # Split B1 into four 2x2 blocks:
    # B11: cols 0-1 (indices 0-3)
    # B12: cols 2-3 (indices 4-7)
    # B13: cols 4-5 (indices 8-11)
    # B14: cols 6-7 (indices 12-15)
    # C1 blocks:
    # C11: cols 0-1 (indices 0-3)
    # C12: cols 2-3 (indices 4-7)
    # C13: cols 4-5 (indices 8-11)
    # C14: cols 6-7 (indices 12-15)
    
    add_strassen(0, 0, 0, 0)    # A1 * B11 -> C11
    add_strassen(0, 4, 4, 7)    # A1 * B12 -> C12
    add_strassen(0, 8, 8, 14)   # A1 * B13 -> C13
    add_strassen(0, 12, 12, 21) # A1 * B14 -> C14
    
    # Part 2: A2 (2x2) * B2 (2x8)
    # A2 is rows 2-3 of A (indices 4-7)
    # B2 is rows 2-3 of B (indices 16-31)
    # C2 is rows 2-3 of C (indices 8-15)
    # B2 blocks:
    # B21: cols 0-1 (indices 16-19)
    # B22: cols 2-3 (indices 20-23)
    # B23: cols 4-5 (indices 24-27)
    # B24: cols 6-7 (indices 28-31)
    # C2 blocks:
    # C21: cols 0-1 (indices 8-11)
    # C22: cols 2-3 (indices 12-15)
    # C23: cols 4-5 (indices 16-19)
    # C24: cols 6-7 (indices 20-23)
    
    add_strassen(4, 16, 8, 28)   # A2 * B21 -> C21
    add_strassen(4, 20, 12, 35)  # A2 * B22 -> C22
    add_strassen(4, 24, 16, 42)  # A2 * B23 -> C23
    add_strassen(4, 28, 20, 49)  # A2 * B24 -> C24

    return {"U": U, "V": V, "W": W}
