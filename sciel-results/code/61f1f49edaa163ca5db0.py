import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 4
    
    # Rank 11 decomposition for 3x2x2
    # A: 3x2, B: 2x2, C: 3x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen for rows 0,1 of A
    # M1 = (A00+A11)*(B00+B11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3 = A00*(B10-B00)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4 = A11*(B01-B11)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Trivial for row 2
    # M8 = A20*B00
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9 = A20*B01
    U_loc[4, 8] = 1
    V_loc[1, 8] = 1
    W_loc[5, 8] = 1
    
    # M10 = A21*B10
    U_loc[5, 9] = 1
    V_loc[2, 9] = 1
    W_loc[4, 9] = 1
    
    # M11 = A21*B11
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
                
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verification
    if time.time() < deadline:
        # Check a few random points or all?
        # All points is 12*16 = 192 checks.
        # Each check is sum of 44 products.
        # 192 * 44 = 8448 operations. Very fast.
        
        valid = True
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col != b_row:
                    continue
                c_row = a_row
                c_col = b_col
                c = c_col * n + c_row
                expected = 1.0
                computed = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(computed - expected) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
             # Fallback to trivial
             U_fallback = np.zeros((n*m, n*m*p))
             V_fallback = np.zeros((m*p, n*m*p))
             W_fallback = np.zeros((p*n, n*m*p))
             for a_idx in range(n*m):
                 a_r, a_c = divmod(a_idx, m)
                 for b_idx in range(m*p):
                     b_r, b_c = divmod(b_idx, p)
                     if a_c == b_r:
                         c_idx = b_c * n + a_r
                         rank_idx = a_idx * (m*p) + b_idx
                         U_fallback[a_idx, rank_idx] = 1
                         V_fallback[b_idx, rank_idx] = 1
                         W_fallback[c_idx, rank_idx] = 1
             return {
                 "U": U_fallback.tolist(),
                 "V": V_fallback.tolist(),
                 "W": W_fallback.tolist()
             }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
