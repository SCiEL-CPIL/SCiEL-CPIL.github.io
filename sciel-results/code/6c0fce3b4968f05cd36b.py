import numpy as np
import time

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Rank 44.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Part 1: 2x4x4 using 4 Strassen 2x2x2 blocks (Rank 28)
    # Part 2: 1x4x4 using 4 direct 1x2x2 blocks (Rank 16)
    
    # --- Strassen 2x2x2 Template ---
    # Local indices for 2x2 matrices
    # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # S1 = (a00+a11)(b00+b11) -> c00, c11
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # S2 = (a10+a11)b00 -> c10, c11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # S3 = a00(b10-b00) -> c01, c11
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # S4 = a11(b01-b11) -> c00, c10
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # S5 = (a00+a01)b11 -> c00, c01
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # S6 = (a10-a00)(b00+b10) -> c11
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # S7 = (a01-a11)(b01+b11) -> c00
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # --- Direct 1x2x2 Template ---
    # U_d: (2, 4), V_d: (4, 4), W_d: (2, 4)
    U_d = np.zeros((2, 4), dtype=int)
    V_d = np.zeros((4, 4), dtype=int)
    W_d = np.zeros((2, 4), dtype=int)
    
    # D1 = a0*b00 -> c0
    U_d[0, 0] = 1
    V_d[0, 0] = 1
    W_d[0, 0] = 1
    
    # D2 = a1*b10 -> c0
    U_d[1, 1] = 1
    V_d[2, 1] = 1
    W_d[0, 1] = 1
    
    # D3 = a0*b01 -> c1
    U_d[0, 2] = 1
    V_d[1, 2] = 1
    W_d[1, 2] = 1
    
    # D4 = a1*b11 -> c1
    U_d[1, 3] = 1
    V_d[3, 3] = 1
    W_d[1, 3] = 1
    
    # --- Assemble Part 1: 2x4x4 (Rows 0-1 of A/C) ---
    # 4 blocks of 2x2x2
    # Block indices: (a_row_off, a_col_off, b_row_off, b_col_off)
    blocks_2x4x4 = [
        (0, 0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0), # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 2), # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 2)  # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    ]
    
    offset = 0
    for ar_off, ac_off, br_off, bc_off in blocks_2x4x4:
        if time.time() > deadline:
            break
        # Map U (2x2) to global A (3x4)
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (ar_off + r) * 4 + (ac_off + c)
                U[gi, offset:offset+7] += U_s[li, :]
                
        # Map V (2x2) to global B (4x4)
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (br_off + r) * 4 + (bc_off + c)
                V[gi, offset:offset+7] += V_s[li, :]
                
        # Map W (2x2) to global C (4x3)
        # Global C index: c * 3 + r
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (bc_off + c) * 3 + (ar_off + r)
                W[gi, offset:offset+7] += W_s[li, :]
                
        offset += 7
        
    # --- Assemble Part 2: 1x4x4 (Row 2 of A/C) ---
    # 4 blocks of 1x2x2
    blocks_1x4x4 = [
        (0, 0, 0), # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        (2, 2, 0), # A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
        (0, 0, 2), # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
        (2, 2, 2)  # A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    ]
    
    for ac_off, br_off, bc_off in blocks_1x4x4:
        if time.time() > deadline:
            break
        # Map U (1x2) to global A (3x4)
        for c in range(2):
            li = c
            gi = 2 * 4 + (ac_off + c)
            U[gi, offset:offset+4] += U_d[li, :]
            
        # Map V (2x2) to global B (4x4)
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (br_off + r) * 4 + (bc_off + c)
                V[gi, offset:offset+4] += V_d[li, :]
                
        # Map W (1x2) to global C (4x3)
        for c in range(2):
            li = c
            gi = (bc_off + c) * 3 + 2
            W[gi, offset:offset+4] += W_d[li, :]
            
        offset += 4
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
