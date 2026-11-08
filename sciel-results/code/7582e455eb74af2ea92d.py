import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists
    # Rank is 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_block(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1):
        nonlocal col_idx
        # A indices
        a_00 = a_i0 * m + a_j0
        a_01 = a_i0 * m + a_j1
        a_10 = a_i1 * m + a_j0
        a_11 = a_i1 * m + a_j1
        
        # B indices
        b_00 = b_j0 * p + b_k0
        b_01 = b_j0 * p + b_k1
        b_10 = b_j1 * p + b_k0
        b_11 = b_j1 * p + b_k1
        
        # C indices
        c_00 = b_k0 * n + a_i0
        c_01 = b_k1 * n + a_i0
        c_10 = b_k0 * n + a_i1
        c_11 = b_k1 * n + a_i1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a_00][col_idx] += 1
        U[a_11][col_idx] += 1
        V[b_00][col_idx] += 1
        V[b_11][col_idx] += 1
        W[c_00][col_idx] += 1
        W[c_11][col_idx] += 1
        col_idx += 1
        
        # M2 = (A10 + A11)B00
        U[a_10][col_idx] += 1
        U[a_11][col_idx] += 1
        V[b_00][col_idx] += 1
        W[c_10][col_idx] += 1
        col_idx += 1
        
        # M3 = A00(B01 - B11)
        U[a_00][col_idx] += 1
        V[b_01][col_idx] += 1
        V[b_11][col_idx] -= 1
        W[c_01][col_idx] += 1
        col_idx += 1
        
        # M4 = A11(B10 - B00)
        U[a_11][col_idx] += 1
        V[b_10][col_idx] += 1
        V[b_00][col_idx] -= 1
        W[c_00][col_idx] += 1
        W[c_10][col_idx] += 1
        col_idx += 1
        
        # M5 = (A00 + A01)B11
        U[a_00][col_idx] += 1
        U[a_01][col_idx] += 1
        V[b_11][col_idx] += 1
        W[c_00][col_idx] -= 1
        W[c_01][col_idx] += 1
        col_idx += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a_10][col_idx] += 1
        U[a_00][col_idx] -= 1
        V[b_00][col_idx] += 1
        V[b_01][col_idx] += 1
        W[c_11][col_idx] += 1
        col_idx += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a_01][col_idx] += 1
        U[a_11][col_idx] -= 1
        V[b_10][col_idx] += 1
        V[b_11][col_idx] += 1
        W[c_00][col_idx] += 1
        W[c_11][col_idx] += 1
        col_idx += 1

    def add_schoolbook_block(a_i0, a_i1, a_j, b_j, b_k0, b_k1):
        nonlocal col_idx
        a_0 = a_i0 * m + a_j
        a_1 = a_i1 * m + a_j
        b_0 = b_j * p + b_k0
        b_1 = b_j * p + b_k1
        c_00 = b_k0 * n + a_i0
        c_01 = b_k1 * n + a_i0
        c_10 = b_k0 * n + a_i1
        c_11 = b_k1 * n + a_i1
        
        # C00 = A0 B0
        U[a_0][col_idx] += 1
        V[b_0][col_idx] += 1
        W[c_00][col_idx] += 1
        col_idx += 1
        
        # C01 = A0 B1
        U[a_0][col_idx] += 1
        V[b_1][col_idx] += 1
        W[c_01][col_idx] += 1
        col_idx += 1
        
        # C10 = A1 B0
        U[a_1][col_idx] += 1
        V[b_0][col_idx] += 1
        W[c_10][col_idx] += 1
        col_idx += 1
        
        # C11 = A1 B1
        U[a_1][col_idx] += 1
        V[b_1][col_idx] += 1
        W[c_11][col_idx] += 1
        col_idx += 1

    # Blocks
    # A1: rows 0,1 cols 0,1
    # A2: rows 0,1 cols 2,3
    # A3: rows 0,1 col 4
    
    # B1: rows 0,1,2,3 cols 0,1
    # B2: rows 0,1,2,3 cols 2,3
    # B3: rows 0,1,2,3 cols 4,5
    # B4: row 4 cols 0,1
    # B5: row 4 cols 2,3
    # B6: row 4 cols 4,5
    
    # Strassen Blocks
    # 1. A1 (0,1) x B1 (0,1)
    add_strassen_block(0, 1, 0, 1, 0, 1, 0, 1)
    # 2. A2 (2,3) x B1 (2,3) -> Wait, B1 rows are 0,1,2,3.
    # My previous logic:
    # B1 (cols 0,1) split into rows 0,1 and 2,3.
    # A1 (cols 0,1) matches rows 0,1.
    # A2 (cols 2,3) matches rows 2,3.
    
    # So:
    # 1. A1 (cols 0,1) x B (rows 0,1, cols 0,1)
    add_strassen_block(0, 1, 0, 1, 0, 1, 0, 1)
    # 2. A2 (cols 2,3) x B (rows 2,3, cols 0,1)
    add_strassen_block(0, 1, 2, 3, 2, 3, 0, 1)
    
    # 3. A1 (cols 0,1) x B (rows 0,1, cols 2,3)
    add_strassen_block(0, 1, 0, 1, 0, 1, 2, 3)
    # 4. A2 (cols 2,3) x B (rows 2,3, cols 2,3)
    add_strassen_block(0, 1, 2, 3, 2, 3, 2, 3)
    
    # 5. A1 (cols 0,1) x B (rows 0,1, cols 4,5)
    add_strassen_block(0, 1, 0, 1, 0, 1, 4, 5)
    # 6. A2 (cols 2,3) x B (rows 2,3, cols 4,5)
    add_strassen_block(0, 1, 2, 3, 2, 3, 4, 5)
    
    # Schoolbook Blocks
    # A3 (col 4) x B (row 4, cols 0,1)
    add_schoolbook_block(0, 1, 4, 4, 0, 1)
    # A3 (col 4) x B (row 4, cols 2,3)
    add_schoolbook_block(0, 1, 4, 4, 2, 3)
    # A3 (col 4) x B (row 4, cols 4,5)
    add_schoolbook_block(0, 1, 4, 4, 4, 5)
    
    return {"U": U, "V": V, "W": W}
