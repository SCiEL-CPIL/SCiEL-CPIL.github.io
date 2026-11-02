import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # Try rank-10 decomposition for 3x2x2
    # U: (6, rank), V: (4, rank), W: (6, rank)
    # A indices: 0..5 (row*2+col), B indices: 0..3 (row*2+col), C indices: 0..5 (row*2+col)
    # C output convention: c = c_col*n + c_row (but for local 3x2x2, n_local=3, so c = c_col*3 + c_row)
    
    # Wait, the global W convention is W[k*n+i, r] where k is col, i is row.
    # For local block, C has 3 rows and 2 cols, so local c_idx = c_col*3 + c_row
    
    def try_rank10():
        """Try to construct a rank-10 decomposition for 3x2x2."""
        rank = 10
        U = np.zeros((6, rank))
        V = np.zeros((4, rank))
        W = np.zeros((6, rank))  # 6 = 2 cols * 3 rows
        
        # Strassen for top 2x2x2 (rows 0,1 of A; all of B; rows 0,1 of C)
        # M1: (a00+a11)(b00+b11) -> c00, c11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1  # c00, c11 (c_idx = col*3+row: c00=0, c11=1*3+1=4)
        
        # M2: (a10+a11)b00 -> c10, c11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[3, 1] = 1; W[4, 1] = -1  # c10=0*3+1=1... wait
        
        # Let me re-index C properly.
        # C is 3x2: C[i,j] for i in 0..2, j in 0..1
        # Global convention: W[k*n+i] where k=col, i=row
        # So C[i,j] -> index j*3 + i
        # C[0,0]=0, C[1,0]=1, C[2,0]=2, C[0,1]=3, C[1,1]=4, C[2,1]=5
        
        U = np.zeros((6, rank))
        V = np.zeros((4, rank))
        W = np.zeros((6, rank))
        
        # M1: (a00+a11)(b00+b11) -> c00, c11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1  # c[0,0]=0, c[1,1]=4
        
        # M2: (a10+a11)b00 -> c10, c11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[4, 1] = -1  # c[1,0]=1, c[1,1]=4
        
        # M3: a00(b10-b00) -> c01, c11
        U[0, 2] = 1
        V[2, 2] = 1; V[0, 2] = -1
        W[3, 2] = 1; W[4, 2] = 1  # c[0,1]=3, c[1,1]=4
        
        # M4: a11(b01-b11) -> c00, c10
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1  # c[0,0]=0, c[1,0]=1
        
        # M5: (a00+a01)b11 -> c00, c01
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[3, 4] = 1  # c[0,0]=0, c[0,1]=3
        
        # M6: (a10-a00)(b00+b10) -> c11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[4, 5] = 1  # c[1,1]=4
        
        # M7: (a01-a11)(b01+b11) -> c00
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # c[0,0]=0
        
        # Now for row 2 of A: a20, a21
        # c20 = a20*b00 + a21*b10 (index 2)
        # c21 = a20*b01 + a21*b11 (index 5)
        
        # Try to use 2 more products:
        # M8: a20*(b00+b10) -> contributes to c20
        # M9: a21*(b00+b10) -> contributes to c20
        # Then c20 = M8 + M9 - a20*b10 - a21*b00... not clean
        
        # Alternative: M8 = a20*b00 + a21*b10 directly? That's one product only if we could factor it,
        # but a20*b00 + a21*b10 is not a single product.
        
        # Let me just use 4 trivial products for row 2:
        # M8: a20*b00 -> c20
        U[4, 7] = 1
        V[0, 7] = 1
        W[2, 7] = 1
        
        # M9: a21*b10 -> c20
        U[5, 8] = 1
        V[2, 8] = 1
        W[2, 8] = 1
        
        # M10: a20*b01 -> c21
        U[4, 9] = 1
        V[1, 9] = 1
        W[5, 9] = 1
        
        # That's only 3 extra, but I have 10 slots (0-9). Let me recount.
        # I used indices 0-9, that's 10. But I only defined up to index 9 with 4 trivial.
        # Wait: 0,1,2,3,4,5,6 = 7 Strassen, then 7,8,9 = 3 trivial. That's 10 total.
        # But I need 4 trivial for row 2. So rank 10 doesn't work with this structure.
        
        return None
    
    # Rank 10 doesn't work with naive extension. Let me stick with rank 11 per block = 44 total,
    # but try to verify it's correct and return it.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication, rank 11
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # C local indexing: C[i,j] -> j*3 + i
    # c00=0, c10=1, c20=2, c01=3, c11=4, c21=5
    
    # Strassen part (cols 0-6)
    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[4, 0] = 1  # c00, c11
    
    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[1, 1] = 1; W_loc[4, 1] = -1  # c10, c11
    
    # M3
    U_loc[0, 2] = 1
    V_loc[2, 2] = 1; V_loc[0, 2] = -1
    W_loc[3, 2] = 1; W_loc[4, 2] = 1  # c01, c11
    
    # M4
    U_loc[3, 3] = 1
    V_loc[1, 3] = 1; V_loc[3, 3] = -1
    W_loc[0, 3] = 1; W_loc[1, 3] = 1  # c00, c10
    
    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[3, 4] = 1  # c00, c01
    
    # M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[2, 5] = 1
    W_loc[4, 5] = 1  # c11
    
    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[1, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1  # c00
    
    # Extra part (cols 7-10) for Row 2 of A
    # M8 = a20 * b00 -> c20 (index 2)
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[2, 7] = 1
    
    # M9 = a21 * b10 -> c20 (index 2)
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[2, 8] = 1
    
    # M10 = a20 * b01 -> c21 (index 5)
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11 = a21 * b11 -> c21 (index 5)
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Mapping function
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
                local_c_idx = c * 3 + r  # c_idx = col*3 + row
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    # Block 0: A cols 0-1 * B rows 0-1, cols 0-1 -> C cols 0-1
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A cols 2-3 * B rows 2-3, cols 0-1 -> C cols 0-1
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A cols 0-1 * B rows 0-1, cols 2-3 -> C cols 2-3
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A cols 2-3 * B rows 2-3, cols 2-3 -> C cols 2-3
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        
        a_row, a_col = divmod(a, m)
        b_row, b_col = divmod(b, p)
        c_col, c_row = divmod(c, n)
        
        expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
        computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        
        if abs(computed - expected) > 1e-10:
            # Fallback to trivial rank-48
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
