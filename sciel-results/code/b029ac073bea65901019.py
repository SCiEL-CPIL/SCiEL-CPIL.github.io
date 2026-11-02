import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # Indices in A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # Indices in B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # Indices in C: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    # We define the Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add these to the global U, V, W.
    # The block A1 is at rows 0,1 of A (cols 0,1).
    # The block A2 is at rows 0,1 of A (cols 2,3).
    # B1 is at rows 0,1 of B (cols 0..5).
    # B2 is at rows 2,3 of B (cols 0..5).
    
    # We iterate over the 6 blocks (2 for A1, 3 for B1; 2 for A2, 3 for B2)
    # Actually, 3 blocks for A1B1 and 3 for A2B2.
    
    def add_strassen_block(a_offset, b_offset, c_offset, start_col):
        # a_offset: (row_offset, col_offset) for A
        # b_offset: (row_offset, col_offset) for B
        # c_offset: (row_offset, col_offset) for C
        # start_col: column index in U,V,W to start writing
        
        a_r, a_c = a_offset
        b_r, b_c = b_offset
        c_r, c_c = c_offset
        
        # Local indices for A (2x2)
        # (0,0) -> a_r*4 + a_c
        # (0,1) -> a_r*4 + a_c + 1
        # (1,0) -> a_r*4 + a_c + 2
        # (1,1) -> a_r*4 + a_c + 3
        
        # Local indices for B (2x2)
        # (0,0) -> b_r*6 + b_c
        # (0,1) -> b_r*6 + b_c + 1
        # (1,0) -> b_r*6 + b_c + 2
        # (1,1) -> b_r*6 + b_c + 3
        
        # Local indices for C (2x2)
        # (0,0) -> c_r*2 + c_c
        # (0,1) -> c_r*2 + c_c + 1
        # (1,0) -> c_r*2 + c_c + 2
        # (1,1) -> c_r*2 + c_c + 3
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        U[a_r*4 + a_c][start_col] = 1
        U[a_r*4 + a_c + 3][start_col] = 1
        V[b_r*6 + b_c][start_col] = 1
        V[b_r*6 + b_c + 3][start_col] = 1
        W[c_r*2 + c_c][start_col] = 1
        W[c_r*2 + c_c + 3][start_col] = 1
        
        # Term 2: (A10+A11)B00
        U[a_r*4 + a_c + 2][start_col+1] = 1
        U[a_r*4 + a_c + 3][start_col+1] = 1
        V[b_r*6 + b_c][start_col+1] = 1
        W[c_r*2 + c_c + 2][start_col+1] = 1 # C10
        W[c_r*2 + c_c + 3][start_col+1] = -1 # C11
        
        # Term 3: A00(B01-B11)
        U[a_r*4 + a_c][start_col+2] = 1
        V[b_r*6 + b_c + 1][start_col+2] = 1
        V[b_r*6 + b_c + 3][start_col+2] = -1
        W[c_r*2 + c_c + 1][start_col+2] = 1 # C01
        W[c_r*2 + c_c + 3][start_col+2] = 1 # C11
        
        # Term 4: A11(B10-B00)
        U[a_r*4 + a_c + 3][start_col+3] = 1
        V[b_r*6 + b_c + 2][start_col+3] = 1
        V[b_r*6 + b_c][start_col+3] = -1
        W[c_r*2 + c_c][start_col+3] = 1 # C00
        W[c_r*2 + c_c + 2][start_col+3] = 1 # C10
        
        # Term 5: (A00+A01)B11
        U[a_r*4 + a_c][start_col+4] = 1
        U[a_r*4 + a_c + 1][start_col+4] = 1
        V[b_r*6 + b_c + 3][start_col+4] = 1
        W[c_r*2 + c_c][start_col+4] = -1 # C00
        W[c_r*2 + c_c + 1][start_col+4] = 1 # C01
        
        # Term 6: (A10-A00)(B00+B01)
        U[a_r*4 + a_c + 2][start_col+5] = 1
        U[a_r*4 + a_c][start_col+5] = -1
        V[b_r*6 + b_c][start_col+5] = 1
        V[b_r*6 + b_c + 1][start_col+5] = 1
        W[c_r*2 + c_c + 3][start_col+5] = 1 # C11
        
        # Term 7: (A01-A11)(B10+B11)
        U[a_r*4 + a_c + 1][start_col+6] = 1
        U[a_r*4 + a_c + 3][start_col+6] = -1
        V[b_r*6 + b_c + 2][start_col+6] = 1
        V[b_r*6 + b_c + 3][start_col+6] = 1
        W[c_r*2 + c_c][start_col+6] = 1 # C00
        W[c_r*2 + c_c + 3][start_col+6] = -1 # C11
        
        return start_col + 7

    # Block A1B1
    # A1 is at (0,0)
    # B1 is split into 3 blocks of width 2: (0,0), (0,2), (0,4)
    # C is at (0,0), (0,2), (0,4)
    
    col = add_strassen_block((0,0), (0,0), (0,0), col)
    col = add_strassen_block((0,0), (0,2), (0,2), col)
    col = add_strassen_block((0,0), (0,4), (0,4), col)
    
    # Block A2B2
    # A2 is at (0,2)
    # B2 is at rows 2,3. Split into 3 blocks: (2,0), (2,2), (2,4)
    # C is at (0,0), (0,2), (0,4) (same as A1B1, so they add up)
    
    col = add_strassen_block((0,2), (2,0), (0,0), col)
    col = add_strassen_block((0,2), (2,2), (0,2), col)
    col = add_strassen_block((0,2), (2,4), (0,4), col)
    
    return {"U": U, "V": V, "W": W}
