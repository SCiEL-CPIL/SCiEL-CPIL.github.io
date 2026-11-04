import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U_vecs, V_vecs, W_vecs for the 7 products
    # Indices in 2x2 block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A block indices: 0,1,2,3
    # B block indices: 0,1,2,3
    # C block indices: 0,1,2,3
    
    # M1: (a00+a11)(b00+b11) -> c00, c01, c10, c11
    # Wait, Strassen output mapping:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M1
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 0, 1], # M1: a00, a11
        [0, 0, 1, 1], # M2: a10, a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00, a01
        [-1, 0, 1, 0],# M6: -a00, a10
        [0, 1, -1, 0] # M7: a01, -a11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: b00, b11
        [1, 0, 0, 0], # M2: b00
        [-1, 0, 1, 0],# M3: -b00, b10
        [0, 1, -1, 0],# M4: b01, -b11
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00, b01
        [0, 0, 1, 1]  # M7: b10, b11
    ]
    
    strassen_W = [
        [1, 1, 0, 0], # M1: c00, c01 (Wait, c00 gets M1, c01 gets M1)
        [0, 0, 1, -1],# M2: c10, -c11
        [0, 1, 0, 1], # M3: c01, c11
        [1, 0, 1, 0], # M4: c00, c10
        [-1, 0, 0, 0],# M5: -c00
        [0, 0, 0, 1], # M6: c11
        [1, 0, 0, 0]  # M7: c00
    ]
    
    # 8 blocks:
    # 0: A_L, B_11 -> K1
    # 1: A_L, B_12 -> K2
    # 2: A_R, B_21 -> K3
    # 3: A_R, B_22 -> K4
    # 4: A_L, C_11 -> K5
    # 5: A_L, C_12 -> K6
    # 6: A_R, C_21 -> K7
    # 7: A_R, C_22 -> K8
    
    # A is 2x4. A_L is cols 0,1. A_R is cols 2,3.
    # B is 4x8. B_11 is rows 0,1 cols 0,1. B_12 is rows 0,1 cols 2,3.
    # B_21 is rows 2,3 cols 0,1. B_22 is rows 2,3 cols 2,3.
    # C_11 is rows 0,1 cols 4,5. C_12 is rows 0,1 cols 6,7.
    # C_21 is rows 2,3 cols 4,5. C_22 is rows 2,3 cols 6,7.
    
    # Block definitions: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    # A is 2x4. Rows 0,1.
    # B is 4x8.
    # C is 2x8.
    
    blocks = [
        # (A_col_start, B_row_start, B_col_start, C_col_start)
        # A_L (cols 0,1), B_11 (rows 0,1, cols 0,1) -> C col 0? No.
        # C = A B.
        # C_1 = A B_1 (2x4). C_1 = [K1+K3, K2+K4].
        # C_1 col 0,1 correspond to K1+K3.
        # K1 = A_L B_11.
        # K3 = A_R B_21.
        # So C_1[:, 0:2] = K1 + K3.
        # C_1[:, 2:4] = K2 + K4.
        # C_2 = A B_2 (2x4). C_2 = [K5+K7, K6+K8].
        # C_2 col 0,1 correspond to K5+K7.
        # K5 = A_L C_11 (B_2 top-left).
        # K7 = A_R C_21 (B_2 bottom-left).
        # So C_2[:, 0:2] = K5 + K7.
        # C_2[:, 2:4] = K6 + K8.
        
        # Global C cols:
        # C_1 is cols 0-3. C_2 is cols 4-7.
        # C_1[:, 0:2] = K1 + K3.
        # C_1[:, 2:4] = K2 + K4.
        # C_2[:, 0:2] (global 4:6) = K5 + K7.
        # C_2[:, 2:4] (global 6:8) = K6 + K8.
        
        # Block 0: K1. A_L (cols 0,1), B_11 (rows 0,1, cols 0,1). Output to C cols 0,1.
        (0, 0, 0, 0), # A_col_start=0, B_row_start=0, B_col_start=0, C_col_start=0
        # Block 1: K2. A_L (cols 0,1), B_12 (rows 0,1, cols 2,3). Output to C cols 2,3.
        (0, 0, 2, 2),
        # Block 2: K3. A_R (cols 2,3), B_21 (rows 2,3, cols 0,1). Output to C cols 0,1.
        (2, 2, 0, 0),
        # Block 3: K4. A_R (cols 2,3), B_22 (rows 2,3, cols 2,3). Output to C cols 2,3.
        (2, 2, 2, 2),
        # Block 4: K5. A_L (cols 0,1), C_11 (rows 0,1, cols 4,5). Output to C cols 4,5.
        (0, 0, 4, 4),
        # Block 5: K6. A_L (cols 0,1), C_12 (rows 0,1, cols 6,7). Output to C cols 6,7.
        (0, 0, 6, 6),
        # Block 6: K7. A_R (cols 2,3), C_21 (rows 2,3, cols 4,5). Output to C cols 4,5.
        (2, 2, 4, 4),
        # Block 7: K8. A_R (cols 2,3), C_22 (rows 2,3, cols 6,7). Output to C cols 6,7.
        (2, 2, 6, 6)
    ]
    
    col_idx = 0
    for a_col_start, b_row_start, b_col_start, c_col_start in blocks:
        for k in range(7):
            # U
            for r in range(2):
                for c in range(2):
                    u_val = strassen_U[k][r*2+c]
                    if u_val != 0:
                        U[r*m + (a_col_start+c)][col_idx] += u_val
            
            # V
            for r in range(2):
                for c in range(2):
                    v_val = strassen_V[k][r*2+c]
                    if v_val != 0:
                        V[(b_row_start+r)*p + (b_col_start+c)][col_idx] += v_val
            
            # W
            for r in range(2):
                for c in range(2):
                    w_val = strassen_W[k][r*2+c]
                    if w_val != 0:
                        W[(c_col_start+c)*n + r][col_idx] += w_val
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
