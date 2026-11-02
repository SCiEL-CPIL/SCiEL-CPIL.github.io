import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
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
    
    # Strassen part (cols 0-6)
    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1 # c00, c11
    
    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1 # c10, c11
    
    # M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1 # c01, c11
    
    # M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1 # c00, c10
    
    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1 # c00, c01
    
    # M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1 # c11
    
    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1 # c00
    
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
    
    # Mapping function
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # block_idx: 0..3
        # a_row_start: 0..2 (always 0 for 3x2)
        # a_col_start: 0 or 2
        # b_row_start: 0 or 2
        # b_col_start: 0 or 2
        # c_row_start: 0..2 (always 0)
        # c_col_start: 0 or 2
        
        col_offset = block_idx * 11
        
        for r in range(3): # Local A rows
            for c in range(2): # Local A cols
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                
                # Copy U_loc
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
                
        for r in range(2): # Local B rows
            for c in range(2): # Local B cols
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                
                # Copy V_loc
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
                
        for r in range(3): # Local C rows
            for c in range(2): # Local C cols
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                
                # Copy W_loc
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    # Block 0: A_L (cols 0,1) * B_11 (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # A_L is rows 0-2, cols 0-1
    # B_11 is rows 0-1, cols 0-1
    # C_L is rows 0-2, cols 0-1
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A_R (cols 2,3) * B_21 (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # A_R is rows 0-2, cols 2-3
    # B_21 is rows 2-3, cols 0-1
    # C_L is rows 0-2, cols 0-1
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A_L (cols 0,1) * B_12 (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # A_L is rows 0-2, cols 0-1
    # B_12 is rows 0-1, cols 2-3
    # C_R is rows 0-2, cols 2-3
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A_R (cols 2,3) * B_22 (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    # A_R is rows 0-2, cols 2-3
    # B_22 is rows 2-3, cols 2-3
    # C_R is rows 0-2, cols 2-3
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify decomposition is correct
    if time.time() < deadline:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check: for random A, B, verify C = A*B matches the decomposition
        # C[i,j] = sum_r U[i,j,r] * V[r] * W[r] where V[r] = sum_k U[k,r]*B[k,r]... 
        # Actually: C = sum_r outer(U[:,r], V[:,r]) * W[:,r]... no.
        # The decomposition is: C[i,j] = sum_r U[i,j,r] * (sum_k A[k,r] * B[...]) 
        # Standard: C = sum_r U_r ⊗ V_r ⊗ W_r where (U_r ⊗ V_r ⊗ W_r)[i,j] = U[i,r]*V[j,r]*W... 
        # Let me just verify with the tensor definition.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r] should equal the matrix multiplication tensor.
        
        # Build the tensor from U,V,W
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # The matrix multiplication tensor: T_mm[a,b,c] = 1 if a_col == b_row and c_row == a_row and c_col == b_col
        # where a = a_row*m + a_col, b = b_row*p + b_col, c = c_col*n + c_row
        
        # Verify a few entries
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            
            a_row, a_col = divmod(a, m)
            b_row, b_col = divmod(b, p)
            c_col, c_row = divmod(c, n)
            
            # Expected: 1 if a_col==b_row and c_row==a_row and c_col==b_col
            expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
            
            # Computed: sum_r U[a,r]*V[b,r]*W[c,r]
            computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
            
            if abs(computed - expected) > 1e-10:
                # Decomposition is incorrect - return trivial rank-48 as fallback
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
