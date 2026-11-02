import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # New approach: partition into blocks and use Strassen for 2x2x2 blocks
    rank = 7 + 7 + 2 + 2  # Strassen for two 2x2x2 blocks, rank-2 for two 1x2x2 blocks
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper for Strassen 2x2x2
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, col_offset):
        # A block: rows a_row_start:a_row_start+2, cols a_col_start:a_col_start+2
        # B block: rows b_row_start:b_row_start+2, cols b_col_start:b_col_start+2
        # C block: rows c_row_start:c_row_start+2, cols c_col_start:c_col_start+2
        
        # Strassen's 7 products for 2x2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        def get_a(r, c):
            return (a_row_start + r) * m + (a_col_start + c)
        
        def get_b(r, c):
            return (b_row_start + r) * p + (b_col_start + c)
        
        def get_c(r, c):
            return (c_col_start + c) * n + (c_row_start + r)
        
        # M1 = (A11 + A22)(B11 + B22)
        col = col_offset
        U[get_a(0, 0), col] = 1
        U[get_a(1, 1), col] = 1
        V[get_b(0, 0), col] = 1
        V[get_b(1, 1), col] = 1
        W[get_c(0, 0), col] = 1
        W[get_c(1, 1), col] = 1
        W[get_c(0, 1), col] = -1
        W[get_c(1, 0), col] = -1
        
        # M2 = (A21 + A22)B11
        col = col_offset + 1
        U[get_a(1, 0), col] = 1
        U[get_a(1, 1), col] = 1
        V[get_b(0, 0), col] = 1
        W[get_c(1, 0), col] = 1
        W[get_c(1, 1), col] = -1
        
        # M3 = A11(B12 - B22)
        col = col_offset + 2
        U[get_a(0, 0), col] = 1
        V[get_b(0, 1), col] = 1
        V[get_b(1, 1), col] = -1
        W[get_c(0, 1), col] = 1
        W[get_c(1, 1), col] = 1
        
        # M4 = A22(B21 - B11)
        col = col_offset + 3
        U[get_a(1, 1), col] = 1
        V[get_b(1, 0), col] = 1
        V[get_b(0, 0), col] = -1
        W[get_c(0, 0), col] = 1
        W[get_c(1, 0), col] = 1
        
        # M5 = (A11 + A12)B22
        col = col_offset + 4
        U[get_a(0, 0), col] = 1
        U[get_a(0, 1), col] = 1
        V[get_b(1, 1), col] = 1
        W[get_c(0, 0), col] = -1
        W[get_c(1, 1), col] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        col = col_offset + 5
        U[get_a(1, 0), col] = 1
        U[get_a(0, 0), col] = -1
        V[get_b(0, 0), col] = 1
        V[get_b(0, 1), col] = 1
        W[get_c(1, 1), col] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        col = col_offset + 6
        U[get_a(0, 1), col] = 1
        U[get_a(1, 1), col] = -1
        V[get_b(1, 0), col] = 1
        V[get_b(1, 1), col] = 1
        W[get_c(0, 0), col] = 1
    
    # Helper for (1x2)x(2x2) -> (1x2) with rank 2
    def add_row_block(a_row, a_col_start, b_row_start, b_col_start, c_row, c_col_start, col_offset):
        # A: 1x2 block at row a_row, cols a_col_start:a_col_start+2
        # B: 2x2 block at rows b_row_start:b_row_start+2, cols b_col_start:b_col_start+2
        # C: 1x2 block at row c_row, cols c_col_start:c_col_start+2
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        
        # We can factor this as:
        # Let X = A[0,0], Y = A[0,1]
        # C[0,0] = X*B[0,0] + Y*B[1,0]
        # C[0,1] = X*B[0,1] + Y*B[1,1]
        
        # This requires rank 2:
        # Factor 1: X * [B[0,0], B[0,1]]
        # Factor 2: Y * [B[1,0], B[1,1]]
        
        col = col_offset
        U[a_row * m + a_col_start, col] = 1
        V[(b_row_start + 0) * p + b_col_start, col] = 1
        V[(b_row_start + 0) * p + b_col_start + 1, col] = 1
        W[(c_col_start + 0) * n + c_row, col] = 1
        W[(c_col_start + 1) * n + c_row, col] = 1
        
        col = col_offset + 1
        U[a_row * m + a_col_start + 1, col] = 1
        V[(b_row_start + 1) * p + b_col_start, col] = 1
        V[(b_row_start + 1) * p + b_col_start + 1, col] = 1
        W[(c_col_start + 0) * n + c_row, col] = 1
        W[(c_col_start + 1) * n + c_row, col] = 1
    
    # Block 1: A[0:1, 0:2] (1x2) x B[0:2, 0:2] (2x2) -> C[0:1, 0:2] (1x2)
    add_row_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:1, 2:4] (1x2) x B[2:4, 0:2] (2x2) -> C[0:1, 2:4] (1x2)
    # Wait, C should be at row 0, cols 2:4? No, C is 3x4, so C[0, 0:2] and C[0, 2:4]
    # Actually, C[0, 0:2] comes from A[0, 0:4] * B[0:4, 0:2]
    # So I need to be more careful with the partitioning.
    
    # Let me redo the partitioning:
    # A is 3x4, B is 4x4, C is 3x4
    # Partition A by rows: [1, 2], by cols: [2, 2]
    # Partition B by rows: [2, 2], by cols: [2, 2]
    # C is partitioned by rows: [1, 2], by cols: [2, 2]
    
    # C[0:1, 0:2] = A[0:1, 0:2] * B[0:2, 0:2] + A[0:1, 2:4] * B[2:4, 0:2]
    # C[0:1, 2:4] = A[0:1, 0:2] * B[0:2, 2:4] + A[0:1, 2:4] * B[2:4, 2:4]
    # C[1:3, 0:2] = A[1:3, 0:2] * B[0:2, 0:2] + A[1:3, 2:4] * B[2:4, 0:2]
    # C[1:3, 2:4] = A[1:3, 0:2] * B[0:2, 2:4] + A[1:3, 2:4] * B[2:4, 2:4]
    
    # So I need 8 sub-blocks, not 4. Let me recalculate.
    
    # Actually, the current solution uses 4 blocks of (3x2)x(2x2), which is different.
    # Let me stick with the current approach but try to optimize it.
    
    # For now, let me just return the current solution to be safe.
    return get_current_solution()

def get_current_solution():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10)
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
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
