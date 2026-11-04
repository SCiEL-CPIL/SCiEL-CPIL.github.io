import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 36
    
    # U: (n*m, rank) = (6, 36)
    # V: (m*p, rank) = (24, 36)  
    # W: (p*n, rank) = (16, 36)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # Split B[0:2, 0:8] into 4 blocks of 2 columns.
    # Each block is 2x2, use Strassen (rank 7).
    for block in range(4):
        b_off = block * 2
        
        # Strassen 7 terms for 2x2 * 2x2
        # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
        # B indices: B00=(0,b_off), B01=(0,b_off+1), B10=(1,b_off), B11=(1,b_off+1)
        # C indices: C00=(0,b_off), C01=(0,b_off+1), C10=(1,b_off), C11=(1,b_off+1)
        
        a00, a01, a10, a11 = 0, 1, 3, 4  # U indices
        b00 = 0*p + b_off; b01 = 0*p + b_off+1; b10 = 1*p + b_off; b11 = 1*p + b_off+1
        c00 = b_off*n + 0; c01 = (b_off+1)*n + 0; c10 = b_off*n + 1; c11 = (b_off+1)*n + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += -1
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] += -1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1; W[c01, col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8]
    # Direct terms: 2 * 8 = 16
    for i in range(2):
        for k in range(8):
            u_idx = i*m + 2
            v_idx = 2*p + k
            w_idx = k*n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
