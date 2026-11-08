import numpy as np
import time

def construct():
    """
    Matrix multiplication 3x4 by 4x4.
    Uses rank 44 construction: split 4x4 B into four 2x2 blocks,
    split 3x4 A into two 3x2 halves, giving 4 subproblems of
    3x2 * 2x2 -> 3x2, each with rank 11 (7 Strassen + 4 direct).
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    # Local A indices: 0..5 (row * 2 + col)
    # Local B indices: 0..3 (row * 2 + col)
    # Local C indices: 0..5 (row * 2 + col)
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for rows 0,1 of A
    # M1: (a00+a10)(b00+b11) -> c00, c11
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2: (a10+a11)(b00) -> c10, c11
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3: (a00)(b10+b11) -> c01, c11
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4: (a11)(b01-b00) -> c00, c10
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5: (a00+a01)(b11) -> c00, c01
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6: (a01-a10)(b00+b10) -> c11
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7: (a01-a11)(b11+b01) -> c00
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    # M8 = a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9 = a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10 = a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11 = a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Mapping function to place local factors into global
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
    
    # Block 0: A rows 0-2, cols 0-1 * B rows 0-1, cols 0-1 -> C rows 0-2, cols 0-1
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A rows 0-2, cols 2-3 * B rows 2-3, cols 0-1 -> C rows 0-2, cols 0-1
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A rows 0-2, cols 0-1 * B rows 0-1, cols 2-3 -> C rows 0-2, cols 2-3
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A rows 0-2, cols 2-3 * B rows 2-3, cols 2-3 -> C rows 0-2, cols 2-3
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify we're within time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return what we have (shouldn't happen for this fixed construction)
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
