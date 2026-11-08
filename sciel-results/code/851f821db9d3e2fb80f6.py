import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using Strassen's algorithm.
    # First, define the Strassen decomposition for 2x2x2.
    # U, V, W are 4x7 matrices.
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # P1
    U_s[0, 0] = 1 # A00
    U_s[3, 0] = 1 # A11
    V_s[0, 0] = 1 # B00
    V_s[3, 0] = 1 # B11
    W_s[0, 0] = 1 # C00
    W_s[3, 0] = 1 # C11
    
    # P2
    U_s[2, 1] = 1 # A10
    U_s[3, 1] = 1 # A11
    V_s[0, 1] = 1 # B00
    W_s[0, 1] = 1 # C00
    
    # P3
    U_s[0, 2] = 1 # A00
    V_s[1, 2] = 1 # B01
    V_s[3, 2] = -1 # B11
    W_s[0, 2] = 1 # C00
    
    # P4
    U_s[3, 3] = 1 # A11
    V_s[2, 3] = 1 # B10
    V_s[0, 3] = -1 # B00
    W_s[3, 3] = 1 # C11
    
    # P5
    U_s[0, 4] = 1 # A00
    U_s[1, 4] = 1 # A01
    V_s[3, 4] = 1 # B11
    W_s[3, 4] = 1 # C11
    
    # P6
    U_s[2, 5] = 1 # A10
    U_s[0, 5] = -1 # A00
    V_s[0, 5] = 1 # B00
    V_s[1, 5] = 1 # B01
    W_s[2, 5] = 1 # C10
    
    # P7
    U_s[1, 6] = 1 # A01
    U_s[3, 6] = -1 # A11
    V_s[2, 6] = 1 # B10
    V_s[3, 6] = 1 # B11
    W_s[3, 6] = 1 # C11
    
    # Now build the 4x4x8 decomposition.
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11 (each 2x2).
    # B is 4x8, split into 2x4 blocks B00, B01, B10, B11 (each 2x4).
    # Each 2x4 block is split into two 2x2 blocks.
    # So B00 is B00_0, B00_1.
    
    # We need to compute C = A B.
    # C is 4x8, split into 2x4 blocks C00, C01, C10, C11.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each term like A00 B00 is a 2x2 by 2x4 product.
    # We compute it using two Strassen decompositions.
    # One for B00_0 and one for B00_1.
    
    # Total rank will be 8 terms * 14 rank = 112.
    
    rank = 112
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # Helper to add a Strassen decomposition for a specific block.
    # a_row, a_col: top-left index of A block in A (2x2)
    # b_row, b_col: top-left index of B block in B (2x4)
    # c_row, c_col: top-left index of C block in C (2x4)
    # b_col is 0 or 4? No, B is 4x8.
    # B blocks are 2x4.
    # B00 is rows 0-1, cols 0-3.
    # B01 is rows 0-1, cols 4-7.
    # B10 is rows 2-3, cols 0-3.
    # B11 is rows 2-3, cols 4-7.
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, offset):
        # b_c is 0 or 4 (column offset in B)
        # c_c is 0 or 4 (column offset in C)
        
        # We need two Strassen decompositions for the two 2x2 parts of the 2x4 block.
        for part in range(2):
            # part 0: cols 0-1 of the 2x4 block
            # part 1: cols 2-3 of the 2x4 block
            b_col_offset = b_c + part * 2
            c_col_offset = c_c + part * 2
            
            # Add Strassen decomposition
            for r in range(7):
                idx = offset + r
                # U
                for i in range(2):
                    for j in range(2):
                        val = U_s[i*2+j, r]
                        if val != 0:
                            U[(a_r + i)*4 + (a_c + j), idx] += val
                            
                # V
                for j in range(2):
                    for k in range(2):
                        val = V_s[j*2+k, r]
                        if val != 0:
                            V[(b_r + j)*8 + (b_col_offset + k), idx] += val
                            
                # W
                for k in range(2):
                    for i in range(2):
                        val = W_s[k*2+i, r]
                        if val != 0:
                            W[(c_r + k)*4 + (c_c + i), idx] += val
                            
        return offset + 14

    offset = 0
    # C00 = A00 B00 + A01 B10
    offset = add_strassen(0, 0, 0, 0, 0, 0, offset) # A00 B00
    offset = add_strassen(0, 2, 2, 0, 0, 0, offset) # A01 B10
    
    # C01 = A00 B01 + A01 B11
    offset = add_strassen(0, 0, 0, 4, 0, 4, offset) # A00 B01
    offset = add_strassen(0, 2, 2, 4, 0, 4, offset) # A01 B11
    
    # C10 = A10 B00 + A11 B10
    offset = add_strassen(2, 0, 0, 0, 2, 0, offset) # A10 B00
    offset = add_strassen(2, 2, 2, 0, 2, 0, offset) # A11 B10
    
    # C11 = A10 B01 + A11 B11
    offset = add_strassen(2, 0, 0, 4, 2, 4, offset) # A10 B01
    offset = add_strassen(2, 2, 2, 4, 2, 4, offset) # A11 B11
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
