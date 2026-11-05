import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        if time.time() > deadline:
            break
            
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += -1
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] += -1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1; W[c01, col] += 1
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
