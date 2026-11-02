import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Rank 42 decomposition based on Strassen
    rank = 42
    
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    # Strassen 2x2x2 decomposition components
    # M1..M7
    # U_strassen[t][row, col] for A (2x2)
    U_strassen = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    # V_strassen[t][row, col] for B (2x2)
    V_strassen = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    # W_strassen[t][row, col] for C (2x2)
    W_strassen = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ])
    
    col_offset = 0
    
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x6. Split into B_T (rows 0,1) and B_B (rows 2,3).
    # B_T is 2x6, split into 3 blocks of 2x2: B_T1, B_T2, B_T3.
    # B_B is 2x6, split into 3 blocks of 2x2: B_B1, B_B2, B_B3.
    
    # We perform 6 Strassen multiplications:
    # 1. A_L * B_T1
    # 2. A_L * B_T2
    # 3. A_L * B_T3
    # 4. A_R * B_B1
    # 5. A_R * B_B2
    # 6. A_R * B_B3
    
    # Each uses 7 rank terms. Total 42.
    
    for block_idx in range(6):
        if time.time() > deadline:
            break
            
        # Determine which A block and which B block
        # block 0-2: A_L, B_T1-B_T3
        # block 3-5: A_R, B_B1-B_B3
        
        if block_idx < 3:
            a_row_offset = 0
            a_col_offset = 0
            b_row_offset = 0
            b_col_offset = block_idx * 2
        else:
            a_row_offset = 0
            a_col_offset = 2
            b_row_offset = 2
            b_col_offset = (block_idx - 3) * 2
            
        for t in range(7):
            # U
            for lr in range(2):
                for lc in range(2):
                    val = U_strassen[t][lr*2+lc]
                    if val != 0:
                        U[a_row_offset + lr, a_col_offset + lc + col_offset*0] # Wait, U index is global
                        # U index: row = a_row_offset + lr, col = a_col_offset + lc
                        # But we are filling U[:, col_offset]
                        U[a_row_offset + lr, a_col_offset + lc] # This is wrong. U is (dim_A, rank)
                        # U[row, rank_col]
                        # The rank column is col_offset + t? No, col_offset is the base for this block.
                        # Actually, each block consumes 7 columns of U, V, W.
                        # So for block_idx, the columns are block_idx*7 to block_idx*7+6.
                        pass

    # The previous logic was slightly mixed. Let's restart the filling logic properly.
    
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    col_base = 0
    for block_idx in range(6):
        if time.time() > deadline:
            break
            
        if block_idx < 3:
            a_r0, a_c0 = 0, 0
            b_r0, b_c0 = 0, block_idx * 2
            c_r0, c_c0 = 0, block_idx * 2 # C is 2x6. Rows 0,1. Cols block_idx*2, block_idx*2+1
        else:
            a_r0, a_c0 = 0, 2
            b_r0, b_c0 = 2, (block_idx - 3) * 2
            c_r0, c_c0 = 0, (block_idx - 3) * 2
            
        for t in range(7):
            rank_col = col_base + t
            
            for lr in range(2):
                for lc in range(2):
                    u_val = U_strassen[t][lr*2+lc]
                    if u_val != 0:
                        U[a_r0 + lr, a_c0 + lc] # Wait, U is (2*4, 42).
                        # U index: row = a_r0 + lr, col = a_c0 + lc? No.
                        # U is a matrix of size (n*m, rank).
                        # The entry U[i, k] multiplies A[i] in the k-th term.
                        # So for term k, U[i, k] is the coefficient of A[i].
                        # A[i] is at row a_r0+lr, col a_c0+lc.
                        # So U[a_r0+lr, a_c0+lc, rank_col]? No, U is 2D.
                        # U is (dim_A, rank). dim_A = 8.
                        # The index of A element (r, c) is r*m + c.
                        # So U[a_r0+lr, a_c0+lc] is wrong.
                        # U[(a_r0+lr)*m + (a_c0+lc), rank_col] += u_val
                        
                        U[(a_r0+lr)*m + (a_c0+lc), rank_col] += u_val
                        
                    v_val = V_strassen[t][lr*2+lc]
                    if v_val != 0:
                        # B index: row b_r0+lr, col b_c0+lc
                        V[(b_r0+lr)*p + (b_c0+lc), rank_col] += v_val
                        
                    w_val = W_strassen[t][lr*2+lc]
                    if w_val != 0:
                        # C index: row c_r0+lr, col c_c0+lc
                        # W is (p*n, rank). Index: col*n + row.
                        # C is 2x6.
                        # W index: (c_c0+lc)*n + (c_r0+lr)
                        W[(c_c0+lc)*n + (c_r0+lr), rank_col] += w_val
                        
        col_base += 7

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
