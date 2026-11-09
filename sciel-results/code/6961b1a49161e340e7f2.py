import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 3, 4, 4
    
    # Precompute Strassen 2x2x2 decomposition
    # A = [[a,b],[c,d]], B = [[x,y],[u,v]]
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # a
        [0,  0,  0,  0,  1,  0,  1],  # b
        [0,  1,  0,  0,  0,  1,  0],  # c
        [1,  1,  0,  1,  0,  0, -1],  # d
    ], dtype=float)
    
    V_s = np.array([
        [1,  1,  0,  0,  0,  1,  0],  # x
        [0,  0,  1,  0,  0,  1,  0],  # y
        [1,  0,  0,  1,  0,  0,  1],  # u
        [0,  0, -1, -1,  1,  0,  1],  # v
    ], dtype=float)
    
    W_s = np.array([
        [1,  0,  0,  1, -1,  0,  1],   # C00
        [0,  0,  1,  0,  1,  0, 0],    # C01
        [0,  1,  0,  1,  0,  0, 0],    # C10
        [1, -1, 1,  0,  0, 1,  0],    # C11
    ], dtype=float)
    
    # Initialize U, V, W with zeros
    # U: 12 x 44
    # V: 16 x 44
    # W: 12 x 44
    rank_total = 44
    U = np.zeros((12, rank_total))
    V = np.zeros((16, rank_total))
    W = np.zeros((12, rank_total))
    
    col_offset = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # a_row_start, a_col_start: top-left of A block (2x2)
        # b_row_start, b_col_start: top-left of B block (2x2)
        # c_row_start, c_col_start: top-left of C block (2x2)
        
        for r in range(7):
            # U: 4 rows
            for ai in range(2):
                for aj in range(2):
                    idx_a = a_row_start + ai
                    idx_j = a_col_start + aj
                    u_idx = idx_a * 4 + idx_j
                    U[u_idx, col_offset] += U_s[ai * 2 + aj, r]
            
            # V: 4 rows
            for bj in range(2):
                for bk in range(2):
                    idx_j = b_row_start + bj
                    idx_k = b_col_start + bk
                    v_idx = idx_j * 4 + idx_k
                    V[v_idx, col_offset] += V_s[bj * 2 + bk, r]
            
            # W: 4 rows
            for ck in range(2):
                for ci in range(2):
                    idx_k = c_col_start + ck
                    idx_i = c_row_start + ci
                    w_idx = idx_k * 3 + idx_i
                    W[w_idx, col_offset] += W_s[ck * 2 + ci, r]
            
            col_offset += 1
    
    # Part 1: Top 2 rows of A (2x4) * B (4x4)
    # C_top is 2x4.
    # C_L (cols 0,1) = A_00 B_00 + A_01 B_10
    # C_R (cols 2,3) = A_00 B_01 + A_01 B_11
    
    # A_00: rows 0-1, cols 0-1
    # B_00: rows 0-1, cols 0-1
    # C_L: rows 0-1, cols 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # A_01: rows 0-1, cols 2-3
    # B_10: rows 2-3, cols 0-1
    # C_L: rows 0-1, cols 0-1
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # A_00: rows 0-1, cols 0-1
    # B_01: rows 0-1, cols 2-3
    # C_R: rows 0-1, cols 2-3
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # A_01: rows 0-1, cols 2-3
    # B_11: rows 2-3, cols 2-3
    # C_R: rows 0-1, cols 2-3
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Part 2: Bottom 1 row of A (1x4) * B (4x4)
    # Schoolbook
    for j in range(4):
        for k in range(4):
            u_idx = 2 * 4 + j  # i=2
            v_idx = j * 4 + k
            w_idx = k * 3 + 2  # i=2
            
            U[u_idx, col_offset] = 1.0
            V[v_idx, col_offset] = 1.0
            W[w_idx, col_offset] = 1.0
            col_offset += 1
    
    # Verify
    T_reconstructed = np.zeros((12, 16, 12))
    for r in range(rank_total):
        T_reconstructed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    T_target = np.zeros((12, 16, 12))
    for i in range(3):
        for j in range(4):
            for k in range(4):
                T_target[i*4+j, j*4+k, k*3+i] = 1
                
    if not np.allclose(T_reconstructed, T_target, atol=1e-9):
        # Fallback to schoolbook 48
        U = np.zeros((12, 48))
        V = np.zeros((16, 48))
        W = np.zeros((12, 48))
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1.0
                    V[j*4+k, col] = 1.0
                    W[k*3+i, col] = 1.0
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
