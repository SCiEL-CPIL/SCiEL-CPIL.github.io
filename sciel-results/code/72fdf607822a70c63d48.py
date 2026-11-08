import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before the 150s kill
    
    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n*m, m*p, p*n  # 12, 16, 12
    
    # Build tensor - only the 48 nonzero entries
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
    T = np.zeros((Nm, Mp, Pn))
    for a, b, c in T_indices:
        T[a, b, c] = 1.0
    
    def verify_fast(U, V, W):
        R = U.shape[1]
        if R == 0:
            return False
        # T_rec[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)  # (Nm*Mp, R)
        T_rec_flat = UV_flat @ W.T  # (Nm*Mp, Pn)
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)
    
    def verify_subset(U, V, W, subset_indices):
        """Verify only at specified (a,b,c) indices"""
        R = U.shape[1]
        if R == 0:
            return False
        for a, b, c in subset_indices:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - T[a, b, c]) > 1e-8:
                return False
        return True
    
    # Direct decomposition (rank 48)
    def make_direct():
        R = n * m * p
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    U_dir, V_dir, W_dir = make_direct()
    assert verify_fast(U_dir, V_dir, W_dir)
    
    # Block decomposition (rank 44) - from inspirations
    def build_block_decomp():
        rank = 44
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen for 2x2x2 (rows 0,1 of local A)
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
        
        # Extra for row 2
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1
        
        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1
        
        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1
        
        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1
        
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
        
        return U, V, W
    
    U_blk, V_blk, W_blk = build_block_decomp()
    
    # Verify the block decomposition
    if verify_fast(U_blk, V_blk, W_blk):
        best_U, best_V, best_W = U_blk, V_blk, W_blk
        best_rank = 44
    else:
        best_U, best_V, best_W = U_dir, V_dir, W_dir
        best_rank = 48
    
    # Try to improve: use iterative optimization
    # The equations are: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Improved column removal with efficient verification
    U_try = best_U.copy()
    V_try = best_V.copy()
    W_try = best_W.copy()
    
    # First pass: single column removal
    changed = True
    while changed and time.time() - start_time < timeout - 15:
        changed = False
        R = U_try.shape[1]
        for r in range(R):
            if time.time() - start_time > timeout - 15:
                break
            U_test = np.delete(U_try, r, axis=1)
            V_test = np.delete(V_try, r, axis=1)
            W_test = np.delete(W_try, r, axis=1)
            if verify_fast(U_test, V_test, W_test):
                U_try, V_try, W_try = U_test, V_test, W_test
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                changed = True
                break
    
    # Second pass: try removing pairs of columns simultaneously (limited budget)
    if time.time() - start_time < timeout - 20:
        R = U_try.shape[1]
        max_pairs_checked = 0
        for r1 in range(R):
            if time.time() - start_time > timeout - 20:
                break
            for r2 in range(r1+1, min(r1+5, R)):  # Only check nearby pairs
                if time.time() - start_time > timeout - 20:
                    break
                max_pairs_checked += 1
                mask = np.ones(R, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                U_test = U_try[:, mask]
                V_test = V_try[:, mask]
                W_test = W_try[:, mask]
                if verify_fast(U_test, V_test, W_test):
                    U_try, V_try, W_try = U_test, V_test, W_test
                    best_U, best_V, best_W = U_test, V_test, W_test
                    best_rank -= 2
                    R = U_try.shape[1]
                    break
            if time.time() - start_time > timeout - 20:
                break
    
    # Final verification and cleanup
    # Remove any columns that are all zeros
    if time.time() - start_time < timeout - 5:
        mask = np.any(best_U != 0, axis=0) | np.any(best_V != 0, axis=0) | np.any(best_W != 0, axis=0)
        if np.sum(mask) < best_rank:
            best_U = best_U[:, mask]
            best_V = best_V[:, mask]
            best_W = best_W[:, mask]
            best_rank = best_U.shape[1]
    
    # Final verification
    if not verify_fast(best_U, best_V, best_W):
        # Fall back to direct decomposition
        U_dir, V_dir, W_dir = make_direct()
        best_U, best_V, best_W = U_dir, V_dir, W_dir
        best_rank = 48
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
