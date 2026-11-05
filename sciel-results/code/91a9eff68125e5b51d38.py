import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def strassen_block(U, V, W, a_off, b_row_off, b_col_off, c_col_off, col):
        # A is 2x2 at offset a_off (cols)
        # B is 2x2 at row b_row_off, col b_col_off
        # C is 2x2 at col c_col_off
        
        # Indices for A (2x2)
        a00 = a_off
        a01 = a_off + 1
        a10 = a_off + 4
        a11 = a_off + 5
        
        # Indices for B (2x2)
        b00 = b_row_off * 8 + b_col_off
        b01 = b_row_off * 8 + b_col_off + 1
        b10 = (b_row_off + 1) * 8 + b_col_off
        b11 = (b_row_off + 1) * 8 + b_col_off + 1
        
        # Indices for C (2x2)
        # C is 2x8. Output cols c_col_off, c_col_off+1
        c00 = c_col_off * 2 + 0
        c01 = (c_col_off + 1) * 2 + 0
        c10 = c_col_off * 2 + 1
        c11 = (c_col_off + 1) * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] += -1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1; W[c01, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        return col

    col = 0
    # 4 blocks for top half of B (rows 0,1), 4 blocks for bottom half (rows 2,3)
    # A cols 0,1 for top; 2,3 for bottom
    # B cols 0,1; 2,3; 4,5; 6,7
    
    # Top half: A cols 0,1, B rows 0,1
    for bc in [0, 2, 4, 6]:
        col = strassen_block(U, V, W, 0, 0, bc, bc, col)
        
    # Bottom half: A cols 2,3, B rows 2,3
    for bc in [0, 2, 4, 6]:
        col = strassen_block(U, V, W, 2, 2, bc, bc, col)
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
