import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # m1 to m7
    # U part (X), V part (Y), W part (Z)
    # X indices: 0:00, 1:01, 2:10, 3:11
    # Y indices: 0:00, 1:01, 2:10, 3:11
    # Z indices: 0:00, 1:01, 2:10, 3:11
    
    # Coefficients for X (U)
    # m1: x00+x11 -> [1, 0, 0, 1]
    # m2: x10+x11 -> [0, 0, 1, 1]
    # m3: x00     -> [1, 0, 0, 0]
    # m4: x11     -> [0, 0, 0, 1]
    # m5: x00+x01 -> [1, 1, 0, 0]
    # m6: x10-x00 -> [-1, 0, 1, 0]
    # m7: x01-x11 -> [0, 1, 0, -1]
    
    U_coeff = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Coefficients for Y (V)
    # m1: y00+y11 -> [1, 0, 0, 1]
    # m2: y00     -> [1, 0, 0, 0]
    # m3: y01-y11 -> [0, 1, 0, -1]
    # m4: y10-y00 -> [-1, 0, 1, 0]
    # m5: y11     -> [0, 0, 0, 1]
    # m6: y00+y01 -> [1, 1, 0, 0]
    # m7: y10+y11 -> [0, 0, 1, 1]
    
    V_coeff = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Coefficients for Z (W)
    # z00 = m1 + m4 - m5 + m7
    # z01 = -m3 + m5 + m6
    # z10 = m2 - m4 + m7
    # z11 = m1 - m2 + m3 + m6
    
    # W_coeff[i][j] is coefficient of mj for zi
    # m1 m2 m3 m4 m5 m6 m7
    W_coeff = [
        [1, 0, 0, 1, -1, 0, 1],   # z00
        [0, 0, -1, 0, 1, 1, 0],   # z01
        [0, 1, 0, -1, 0, 0, 1],   # z10
        [1, -1, 1, 0, 0, 1, 0]    # z11
    ]
    
    # Define the 4 sub-problems
    # Each sub-problem: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    # A is 2x4. A_row_offset is 0. A_col_offset is 0 or 2.
    # B is 4x4. B_row_offset is 0 or 2. B_col_offset is 0 or 2.
    # C is 2x4. C_row_offset is 0. C_col_offset is 0 or 2.
    
    sub_problems = [
        # (A_col, B_row, B_col, C_col)
        (0, 0, 0, 0), # A1 B11 -> C_L1
        (2, 2, 0, 0), # A2 B21 -> C_L2
        (0, 0, 2, 2), # A1 B12 -> C_R1
        (2, 2, 2, 2)  # A2 B22 -> C_R2
    ]
    
    col = 0
    for sp in sub_problems:
        a_col = sp[0]
        b_row = sp[1]
        b_col = sp[2]
        c_col = sp[3]
        
        # Local indices for 2x2 matrices
        # X (A part): rows 0,1; cols a_col, a_col+1
        # Global U index: i * m + j
        # i in {0, 1}, j in {a_col, a_col+1}
        # Local index 0: i=0, j=a_col -> Global 0*m + a_col = a_col
        # Local index 1: i=0, j=a_col+1 -> Global a_col+1
        # Local index 2: i=1, j=a_col -> Global 1*m + a_col = 4 + a_col
        # Local index 3: i=1, j=a_col+1 -> Global 5 + a_col
        
        u_indices = [a_col, a_col + 1, 4 + a_col, 5 + a_col]
        
        # Y (B part): rows b_row, b_row+1; cols b_col, b_col+1
        # Global V index: j * p + k
        # j in {b_row, b_row+1}, k in {b_col, b_col+1}
        # Local index 0: j=b_row, k=b_col -> Global b_row*p + b_col
        # Local index 1: j=b_row, k=b_col+1 -> Global b_row*p + b_col+1
        # Local index 2: j=b_row+1, k=b_col -> Global (b_row+1)*p + b_col
        # Local index 3: j=b_row+1, k=b_col+1 -> Global (b_row+1)*p + b_col+1
        
        v_indices = [b_row * 4 + b_col, b_row * 4 + b_col + 1, (b_row + 1) * 4 + b_col, (b_row + 1) * 4 + b_col + 1]
        
        # Z (C part): rows 0,1; cols c_col, c_col+1
        # Global W index: k * n + i
        # k in {c_col, c_col+1}, i in {0, 1}
        # Local index 0: i=0, k=c_col -> Global c_col*2 + 0 = 2*c_col
        # Local index 1: i=0, k=c_col+1 -> Global (c_col+1)*2 + 0 = 2*c_col + 2
        # Local index 2: i=1, k=c_col -> Global c_col*2 + 1 = 2*c_col + 1
        # Local index 3: i=1, k=c_col+1 -> Global (c_col+1)*2 + 1 = 2*c_col + 3
        
        w_indices = [2 * c_col, 2 * c_col + 2, 2 * c_col + 1, 2 * c_col + 3]
        
        # Add 7 Strassen terms
        for t in range(7):
            # Update U
            for li in range(4):
                U[u_indices[li]][col] += U_coeff[t][li]
            
            # Update V
            for li in range(4):
                V[v_indices[li]][col] += V_coeff[t][li]
            
            # Update W
            for li in range(4):
                W[w_indices[li]][col] += W_coeff[li][t]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

