import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U coeffs: A factors
    # V coeffs: B factors
    # W coeffs: C factors
    
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 - b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    u_coeffs = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {2: 1, 0: -1},      # P6
        {1: 1, 3: -1}       # P7
    ]
    
    v_coeffs = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1}        # P7
    ]
    
    w_coeffs = [
        {0: 1, 3: 1},       # P1 -> C00, C11
        {2: 1, 3: -1},      # P2 -> C10, C11
        {1: 1, 3: 1},       # P3 -> C01, C11
        {0: 1, 2: 1},       # P4 -> C00, C10
        {0: -1, 1: 1},      # P5 -> C00, C01
        {3: 1},             # P6 -> C11
        {0: 1}              # P7 -> C00
    ]

    col = 0
    # Iterate over the 6 blocks of 2x2 multiplications
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x6. Split into 3 blocks of 2x2 cols: B0 (cols 0,1), B1 (cols 2,3), B2 (cols 4,5).
    # Each B block is split into B_j_L (rows 0,1) and B_j_R (rows 2,3).
    
    for side in range(2): # 0: L, 1: R
        # A columns offset
        a_col_offset = 2 * side
        # B rows offset
        b_row_offset = 2 * side
        
        for k_block in range(3): # 0, 1, 2
            # B columns offset
            b_col_offset = 2 * k_block
            
            for s in range(7):
                # Fill U (A part)
                for local_idx, coeff in u_coeffs[s].items():
                    # local_idx 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
                    a_row = local_idx // 2
                    a_col_local = local_idx % 2
                    global_a_row = a_row
                    global_a_col = a_col_offset + a_col_local
                    u_idx = global_a_row * m + global_a_col
                    U[u_idx][col] += coeff
                
                # Fill V (B part)
                for local_idx, coeff in v_coeffs[s].items():
                    b_row_local = local_idx // 2
                    b_col_local = local_idx % 2
                    global_b_row = b_row_offset + b_row_local
                    global_b_col = b_col_offset + b_col_local
                    v_idx = global_b_row * p + global_b_col
                    V[v_idx][col] += coeff
                
                # Fill W (C part)
                for local_idx, coeff in w_coeffs[s].items():
                    c_row_local = local_idx // 2
                    c_col_local = local_idx % 2
                    global_c_row = c_row_local
                    global_c_col = b_col_offset + c_col_local
                    w_idx = global_c_col * n + global_c_row
                    W[w_idx][col] += coeff
                
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
