import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 150s kill
    
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    def add_strassen(col_offset, A_idx, B_idx, C_idx):
        """Add 7-term Strassen decomposition for 2x2x2 block multiply.
        A_idx: [A00, A01, A10, A11] row indices in U
        B_idx: [B00, B01, B10, B11] row indices in V
        C_idx: [C00, C01, C10, C11] row indices in W
        """
        a0, a1, a2, a3 = A_idx
        b0, b1, b2, b3 = B_idx
        c0, c1, c2, c3 = C_idx
        
        # P0 = (A00+A11)(B00+B11)
        U[a0][col_offset+0] += 1
        U[a3][col_offset+0] += 1
        V[b0][col_offset+0] += 1
        V[b3][col_offset+0] += 1
        W[c0][col_offset+0] += 1
        W[c3][col_offset+0] += 1
        
        # P1 = (A10+A11)B00
        U[a2][col_offset+1] += 1
        U[a3][col_offset+1] += 1
        V[b0][col_offset+1] += 1
        W[c2][col_offset+1] += 1
        W[c3][col_offset+1] -= 1
        
        # P2 = A00(B11-B01)
        U[a0][col_offset+2] += 1
        V[b1][col_offset+2] += 1
        V[b3][col_offset+2] -= 1
        W[c1][col_offset+2] += 1
        W[c3][col_offset+2] += 1
        
        # P3 = A11(B00-B10)
        U[a3][col_offset+3] += 1
        V[b2][col_offset+3] += 1
        V[b0][col_offset+3] -= 1
        W[c0][col_offset+3] += 1
        W[c2][col_offset+3] += 1
        
        # P4 = (A00+A01)B10
        U[a0][col_offset+4] += 1
        U[a1][col_offset+4] += 1
        V[b3][col_offset+4] += 1
        W[c0][col_offset+4] -= 1
        W[c1][col_offset+4] += 1
        
        # P5 = (A10-A00)B01
        U[a2][col_offset+5] += 1
        U[a0][col_offset+5] -= 1
        V[b0][col_offset+5] += 1
        V[b1][col_offset+5] += 1
        W[c3][col_offset+5] += 1
        
        # P6 = (A01-A11)(B20+B21) -- wait, this doesn't look right
        # Let me use the standard Strassen:
        # P6 = (A01-A11)(B10+B11)
        U[a1][col_offset+6] += 1
        U[a3][col_offset+6] -= 1
        V[b2][col_offset+6] += 1
        V[b3][col_offset+6] += 1
        W[c0][col_offset+6] += 1

    # Define the four 2x2x2 blocks
    # A is 2x4: rows 0,1; columns 0-3
    # A_L = A[:,0:2] -> rows [0,1] in U (indices 0,1 for row0, 4,5 for row1)
    # A_R = A[:,2:4] -> rows [2,3] in U (indices 2,3 for row0, 6,7 for row1)
    
    # B is 4x4: 
    # B_TL = B[0:2,0:2] -> V indices [0,1,4,5]
    # B_TR = B[0:2,2:4] -> V indices [2,3,6,7]
    # B_BL = B[2:4,0:2] -> V indices [8,9,12,13]
    # B_BR = B[2:4,2:4] -> V indices [10,11,14,15]
    
    # C is 2x4:
    # C_L = C[:,0:2] -> W indices [0,1,4,5] (row0: 0,1; row1: 4,5)
    # C_R = C[:,2:4] -> W indices [2,3,6,7] (row0: 2,3; row1: 6,7)
    
    # Block 1: A_L @ B_TL -> C_L
    # A_L: A00=U[0], A01=U[1], A10=U[4], A11=U[5]
    # B_TL: B00=V[0], B01=V[1], B10=V[4], B11=V[5]
    # C_L: C00=W[0], C01=W[1], C10=W[4], C11=W[5]
    add_strassen(0, [0,1,4,5], [0,1,4,5], [0,1,4,5])
    
    # Block 2: A_R @ B_BL -> C_L (add to C_L)
    # A_R: A00=U[2], A01=U[3], A10=U[6], A11=U[7]
    # B_BL: B00=V[8], B01=V[9], B10=V[12], B11=V[13]
    # C_L: C00=W[0], C01=W[1], C10=W[4], C11=W[5]
    add_strassen(7, [2,3,6,7], [8,9,12,13], [0,1,4,5])
    
    # Block 3: A_L @ B_TR -> C_R
    # A_L: A00=U[0], A01=U[1], A10=U[4], A11=U[5]
    # B_TR: B00=V[2], B01=V[3], B10=V[6], B11=V[7]
    # C_R: C00=W[2], C01=W[3], C10=W[6], C11=W[7]
    add_strassen(14, [0,1,4,5], [2,3,6,7], [2,3,6,7])
    
    # Block 4: A_R @ B_BR -> C_R (add to C_R)
    # A_R: A00=U[2], A01=U[3], A10=U[6], A11=U[7]
    # B_BR: B00=V[10], B01=V[11], B10=V[14], B11=V[15]
    # C_R: C00=W[2], C01=W[3], C10=W[6], C11=W[7]
    add_strassen(21, [2,3,6,7], [10,11,14,15], [2,3,6,7])

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
