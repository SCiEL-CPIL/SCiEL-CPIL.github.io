import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    
    # We will use 5 blocks of 2x2 (Strassen rank 7) and 1 block of 2x1 (Standard rank 4)
    # Total rank = 5 * 7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)] # 4 x 39
    V = [[0] * rank for _ in range(m * p)] # 22 x 39
    W = [[0] * rank for _ in range(p * n)] # 22 x 39
    
    col_offset = 0
    
    # Strassen decomposition for 2x2 x 2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # u_s(A) coefficients
    strassen_u = [
        [1, 0, 0, 1], # M1: a00 + a11
        [0, 0, 1, 1], # M2: a10 + a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00 + a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    # v_s(B) coefficients for local B
    strassen_v = [
        [1, 0, 0, 1], # M1: b00 + b11
        [1, 0, 0, 0], # M2: b00
        [0, 0, 1, -1], # M3: b10 - b11
        [-1, 1, 0, 0], # M4: b01 - b00
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00 + b01
        [0, 0, 1, 1]  # M7: b10 + b11
    ]
    
    # w_s(C) coefficients for local C
    strassen_w = [
        [1, 0, 0, -1], # c00: M1 - M4 + M5 - M7 -> coeff of M1 is 1
        [0, 1, 0, 0],   # c01: M3 + M4 -> coeff of M1 is 0
        [0, 0, 0, 0],   # c10: M2 + M7 -> coeff of M1 is 0
        [1, 0, 0, 0]    # c11: M1 - M2 + M3 + M6 -> coeff of M1 is 1
    ]
    # Wait, w_s is for each product s, what is its contribution to each c?
    # Let's define w[s][c_idx]
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M4
    # c10 = M2 + M7
    # c11 = M1 - M2 + M3 + M6
    
    strassen_w = [
        [1, 0, 0, 0], # M1 contributes to c00 (1), c11 (1)
        [0, 0, 0, 0], # M2 contributes to c10 (1), c11 (-1) -> Wait
        # Let's list coefficients for each M_s on each C_ij
        # M1: c00(+1), c11(+1)
        # M2: c10(+1), c11(-1)
        # M3: c01(+1), c11(+1)
        # M4: c00(-1), c01(+1)
        # M5: c00(+1)
        # M6: c11(+1)
        # M7: c00(-1), c10(+1)
        
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [-1, 1, 0, 0], # M4
        [1, 0, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [-1, 0, 1, 0]  # M7
    ]
    
    # Standard decomposition for 2x2 x 1
    # A indices: 0,1,2,3
    # B local indices: 0:(0,0), 1:(1,0)
    # C local indices: 0:(0,0), 1:(1,0)
    
    std_u = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    
    std_v = [
        [1, 0], # b00
        [0, 1], # b10
        [1, 0], # b00
        [0, 1]  # b10
    ]
    
    std_w = [
        [1, 0], # c00
        [0, 1], # c10
        [1, 0], # c00
        [0, 1]  # c10
    ]
    # c00 = M1 + M2
    # c10 = M3 + M4
    
    # Mapping for 2x2 blocks
    for b in range(5):
        # B global indices
        # b00 -> 2b
        # b01 -> 2b+1
        # b10 -> 11+2b
        # b11 -> 11+2b+1
        b_indices = [2*b, 2*b+1, 11+2*b, 11+2*b+1]
        
        # C global indices
        # c00 -> 4b
        # c01 -> 4b+1
        # c10 -> 4b+2
        # c11 -> 4b+3
        c_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
        
        for s in range(7):
            # Fill U
            for i in range(4):
                U[i][col_offset] += strassen_u[s][i]
            
            # Fill V
            for j in range(4):
                V[b_indices[j]][col_offset] += strassen_v[s][j]
            
            # Fill W
            for k in range(4):
                W[c_indices[k]][col_offset] += strassen_w[s][k]
            
            col_offset += 1
    
    # Mapping for 2x1 block (last column)
    b_idx = 10
    b_indices = [2*b_idx, 11+b_idx] # b00, b10
    
    c_indices = [4*b_idx, 4*b_idx+1] # c00, c10
    
    for s in range(4):
        for i in range(4):
            U[i][col_offset] += std_u[s][i]
        
        for j in range(2):
            V[b_indices[j]][col_offset] += std_v[s][j]
        
        for k in range(2):
            W[c_indices[k]][col_offset] += std_w[s][k]
        
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
