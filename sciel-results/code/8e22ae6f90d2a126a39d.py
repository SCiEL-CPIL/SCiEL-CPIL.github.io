import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local indices for the 3x2x2 block
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    # 1. (A11+A22)(B11+B22) -> C11, C22
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # 2. A22 * B11 -> C21, C22 (with sign)
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # 3. A11 * (B22-B12) -> C12, C22
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # 4. (A21-A11) * B22 -> C11, C21
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # 5. (A11+A12) * B21 -> C11, C12 (with sign)
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # 6. A12 * (B11+B21) -> C21
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # 7. (A21+A22) * (B22-B12) -> C11
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part for row 3 (cols 7-10)
    # 8. A31 * B11 -> C31
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    # 9. A32 * B21 -> C31
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    # 10. A31 * B12 -> C32
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    # 11. A32 * B22 -> C32
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
                
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Removing the failing Laderman attempt reduces overhead and ensures the robust rank-44 solution is returned immediately, maintaining the current score while improving reliability.", q: 0.9, delta: [0.0, 0.0]}
