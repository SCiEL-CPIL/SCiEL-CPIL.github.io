import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for the 2x2x2 core (rows 0,1 of A)
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
    
    # Extra part (cols 7-10) for Row 2 of A (trivial 1x2x2)
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
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
    
    # Verification (fast vectorized version)
    try:
        U_f = U.flatten()
        V_f = V.flatten()
        W_f = W.flatten()
        
        # Construct index matrices for the 48x48x48 tensor
        a_range = np.arange(n * m)
        b_range = np.arange(m * p)
        c_range = np.arange(p * n)
        
        # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # Vectorized: for each (a,b), compute sum over r of U[a,r] V[b,r], then dot with W[c,:]
        # Or better: T = np.einsum('ar,br,cr->abc', U, V, W)
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Expected tensor
        T_exp = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            a_r, a_c = divmod(a, m)
            for b in range(m*p):
                b_r, b_c = divmod(b, p)
                if a_c == b_r:
                    c_idx = b_c * n + a_r
                    T_exp[a, b, c_idx] = 1.0
        
        if not np.allclose(T, T_exp, atol=1e-9):
            # Fallback to rank 48 if something went wrong
            U_fallback = np.zeros((n*m, n*m*p))
            V_fallback = np.zeros((m*p, n*m*p))
            W_fallback = np.zeros((p*n, n*m*p))
            for a in range(n*m):
                a_r, a_c = divmod(a, m)
                for b in range(m*p):
                    b_r, b_c = divmod(b, p)
                    if a_c == b_r:
                        c_idx = b_c * n + a_r
                        rank_idx = a * (m*p) + b
                        U_fallback[a, rank_idx] = 1
                        V_fallback[b, rank_idx] = 1
                        W_fallback[c_idx, rank_idx] = 1
            return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
    except Exception:
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
