import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices in local 2x2 space:
    # A: 0,1,2,3 (row-major: A00, A01, A10, A11)
    # B: 0,1,2,3 (row-major: B00, B01, B10, B11)
    # C: 0,1,2,3 (row-major: C00, C01, C10, C11)
    
    # Global indices mapping:
    # A: i*m + j. i in 0..1, j in 0..3.
    # B: j*p + k. j in 0..3, k in 0..7.
    # C: k*n + i. k in 0..7, i in 0..1.
    
    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, base_col):
        # a_row_off, a_col_off: offset in A (2x4)
        # b_row_off, b_col_off: offset in B (4x8)
        # c_row_off, c_col_off: offset in C (2x8)
        
        # Local A indices: 0,1,2,3 -> Global A indices
        # Local B indices: 0,1,2,3 -> Global B indices
        # Local C indices: 0,1,2,3 -> Global C indices
        
        # M1
        # A: A00+A11 (Local 0, 3)
        # B: B00+B11 (Local 0, 3)
        # C: C00, C11
        col = base_col
        U[a_row_off*4 + a_col_off + 0][col] += 1
        U[a_row_off*4 + a_col_off + 3][col] += 1
        V[b_row_off*8 + b_col_off + 0][col] += 1
        V[b_row_off*8 + b_col_off + 3][col] += 1
        W[c_row_off*2 + c_col_off + 0][col] += 1
        W[c_row_off*2 + c_col_off + 3][col] += 1
        
        # M2
        # A: A10+A11 (Local 2, 3)
        # B: B00 (Local 0)
        # C: C10
        col = base_col + 1
        U[a_row_off*4 + a_col_off + 2][col] += 1
        U[a_row_off*4 + a_col_off + 3][col] += 1
        V[b_row_off*8 + b_col_off + 0][col] += 1
        W[c_row_off*2 + c_col_off + 2][col] += 1
        
        # M3
        # A: A00 (Local 0)
        # B: B01-B11 (Local 1, 3)
        # C: C01
        col = base_col + 2
        U[a_row_off*4 + a_col_off + 0][col] += 1
        V[b_row_off*8 + b_col_off + 1][col] += 1
        V[b_row_off*8 + b_col_off + 3][col] -= 1
        W[c_row_off*2 + c_col_off + 1][col] += 1
        
        # M4
        # A: A11 (Local 3)
        # B: B10-B00 (Local 2, 0)
        # C: C00, C10
        col = base_col + 3
        U[a_row_off*4 + a_col_off + 3][col] += 1
        V[b_row_off*8 + b_col_off + 2][col] += 1
        V[b_row_off*8 + b_col_off + 0][col] -= 1
        W[c_row_off*2 + c_col_off + 0][col] += 1
        W[c_row_off*2 + c_col_off + 2][col] += 1
        
        # M5
        # A: A00+A01 (Local 0, 1)
        # B: B11 (Local 3)
        # C: C00, C01
        col = base_col + 4
        U[a_row_off*4 + a_col_off + 0][col] += 1
        U[a_row_off*4 + a_col_off + 1][col] += 1
        V[b_row_off*8 + b_col_off + 3][col] += 1
        W[c_row_off*2 + c_col_off + 0][col] -= 1
        W[c_row_off*2 + c_col_off + 1][col] += 1
        
        # M6
        # A: A10-A00 (Local 2, 0)
        # B: B00+B01 (Local 0, 1)
        # C: C11
        col = base_col + 5
        U[a_row_off*4 + a_col_off + 2][col] += 1
        U[a_row_off*4 + a_col_off + 0][col] -= 1
        V[b_row_off*8 + b_col_off + 0][col] += 1
        V[b_row_off*8 + b_col_off + 1][col] += 1
        W[c_row_off*2 + c_col_off + 3][col] += 1
        
        # M7
        # A: A01-A11 (Local 1, 3)
        # B: B10+B11 (Local 2, 3)
        # C: C00
        col = base_col + 6
        U[a_row_off*4 + a_col_off + 1][col] += 1
        U[a_row_off*4 + a_col_off + 3][col] -= 1
        V[b_row_off*8 + b_col_off + 2][col] += 1
        V[b_row_off*8 + b_col_off + 3][col] += 1
        W[c_row_off*2 + c_col_off + 0][col] += 1

    # Sub-problem 1: A1 B11
    # A offset: 0,0. B offset: 0,0. C offset: 0,0.
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Sub-problem 2: A2 B21
    # A offset: 0,2. B offset: 2,0. C offset: 0,0.
    add_strassen(0, 2, 2, 0, 0, 0, 7)
    
    # Sub-problem 3: A1 B12
    # A offset: 0,0. B offset: 0,2. C offset: 0,2.
    add_strassen(0, 0, 0, 2, 0, 2, 14)
    
    # Sub-problem 4: A2 B22
    # A offset: 0,2. B offset: 2,2. C offset: 0,2.
    add_strassen(0, 2, 2, 2, 0, 2, 21)
    
    return {"U": U, "V": V, "W": W}
