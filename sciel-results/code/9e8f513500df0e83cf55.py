import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 4
    TIME_LIMIT = 115.0
    
    def time_up():
        return (time.time() - start_time) > TIME_LIMIT
    
    def verify(U, V, W, rank):
        """Verify decomposition correctness."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
        
        T_target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def build_strassen():
        """Build the base rank-28 Strassen decomposition."""
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Strassen's 7 multiplications for 2x2 blocks
        # For each 2x2x2 block, we use:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # The 2x4x4 matrix is divided into four 2x2 blocks
        # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
        # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
        # C = [[C11, C12], [C21, C22]] where each Ci is 2x2
        
        # For each block pair, we apply Strassen
        # Block (1,1): A11 * B11 -> C11
        # Block (1,2): A11 * B12 + A12 * B22 -> C12
        # Block (2,1): A21 * B11 + A22 * B21 -> C21
        # Block (2,2): A21 * B12 + A22 * B22 -> C22
        
        # Actually, for 2x4 * 4x4, we have:
        # A is 2x4, B is 4x4, C is 2x4
        # A = [A_left | A_right] where A_left, A_right are 2x2
        # B = [[B_top_left, B_top_right], [B_bot_left, B_bot_right]] each 2x2
        # C = [C_left | C_right] where C_left = A_left * B_top_left + A_right * B_bot_left
        #                                C_right = A_left * B_top_right + A_right * B_bot_right
        
        # So we need to compute:
        # C_left = A_left * B_top_left + A_right * B_bot_left
        # C_right = A_left * B_top_right + A_right * B_bot_right
        
        # Each of these is a sum of two 2x2 matrix products
        # We can use Strassen for each product, but we can also combine
        
        # Let me use the standard approach: treat it as 2 independent 2x2x2 problems
        # for C_left and 2 independent 2x2x2 problems for C_right
        # That gives 4 * 7 = 28 rank
        
        # Strassen steps for 2x2 multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11  
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        def add_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, col_start):
            """Add Strassen decomposition for one 2x2x2 block."""
            # Local indices: for 2x2, index = row*2 + col
            # U: 2x2 = 4 entries, indexed by (i, local_j) where local_j = 0,1
            # V: 2x2 = 4 entries, indexed by (local_j, local_k) 
            # W: 2x2 = 4 entries, indexed by (local_k, i)
            
            # M1 = (A11+A22)(B11+B22)
            # U contributions: A11 (idx 0), A22 (idx 3)
            # V contributions: B11 (idx 0), B22 (idx 3)
            # W contributions: C11 (idx 0), C22 (idx 3)
            col = col_start + 0
            U[a_col_offset + 0][col] += 1  # A11
            U[a_col_offset + 3][col] += 1  # A22
            V[b_row_offset + 0][col] += 1  # B11
            V[b_row_offset + 3][col] += 1  # B22
            W[c_col_offset + 0][col] += 1  # C11
            W[c_col_offset + 3][col] += 1  # C22
            
            # M2 = (A21+A22)B11
            col = col_start + 1
            U[a_col_offset + 2][col] += 1  # A21
            U[a_col_offset + 3][col] += 1  # A22
            V[b_row_offset + 0][col] += 1  # B11
            W[c_col_offset + 2][col] += 1  # C21
            W[c_col_offset + 3][col] -= 1  # C22
            
            # M3 = A11(B21-B22)
            col = col_start + 2
            U[a_col_offset + 0][col] += 1  # A11
            V[b_row_offset + 2][col] += 1  # B21
            V[b_row_offset + 3][col] -= 1  # B22
            W[c_col_offset + 1][col] += 1  # C12
            W[c_col_offset + 3][col] += 1  # C22
            
            # M4 = A22(B12-B11)
            col = col_start + 3
            U[a_col_offset + 3][col] += 1  # A22
            V[b_row_offset + 1][col] += 1  # B12
            V[b_row_offset + 0][col] -= 1  # B11
            W[c_col_offset + 0][col] += 1  # C11
            W[c_col_offset + 2][col] += 1  # C21
            
            # M5 = (A11+A12)B22
            col = col_start + 4
            U[a_col_offset + 0][col] += 1  # A11
            U[a_col_offset + 1][col] += 1  # A12
            V[b_row_offset + 3][col] += 1  # B22
            W[c_col_offset + 0][col] -= 1  # C11
            W[c_col_offset + 1][col] += 1  # C12
            
            # M6 = (A21-A11)(B11+B12)
            col = col_start + 5
            U[a_col_offset + 2][col] += 1  # A21
            U[a_col_offset + 0][col] -= 1  # A11
            V[b_row_offset + 0][col] += 1  # B11
            V[b_row_offset + 1][col] += 1  # B12
            W[c_col_offset + 3][col] += 1  # C22
            
            # M7 = (A12-A22)(B21+B22)
            col = col_start + 6
            U[a_col_offset + 1][col] += 1  # A12
            U[a_col_offset + 3][col] -= 1  # A22
            V[b_row_offset + 2][col] += 1  # B21
            V[b_row_offset + 3][col] += 1  # B22
            W[c_col_offset + 0][col] += 1  # C11
            W[c_col_offset + 1][col] -= 1  # C12
        
        # For 2x4 * 4x4:
        # A has rows 0,1 and cols 0,1,2,3
        # B has rows 0,1,2,3 and cols 0,1,2,3
        # C has rows 0,1 and cols 0,1,2,3
        
        # C[i,0:2] = A[i,0:2] * B[0:2,0:2] + A[i,2:4] * B[2:4,0:2]
        # C[i,2:4] = A[i,0:2] * B[0:2,2:4] + A[i,2:4] * B[2:4,2:4]
        
        # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        add_strassen_block(0, 0, 0, 0, 0)
        # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        add_strassen_block(2, 2, 0, 0, 7)
        # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        add_strassen_block(0, 0, 2, 2, 14)
        # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
        add_strassen_block(2, 2, 2, 2, 21)
        
        return U, V, W, rank
    
    def try_peel(U, V, W, current_rank):
        """
        Try to find a non-trivial linear combination of rank-1 terms that sums to zero.
        If we find coefficients c_r (not all zero) such that:
          sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0
        then we can eliminate one term.
        
        More practically: try to express one term as a linear combination of others.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Try to eliminate each term
        for elim in range(current_rank - 1, -1, -1):
            if time_up():
                return None
            
            other_idx = [j for j in range(current_rank) if j != elim]
            
            # We need: U[:,elim] = U[:,other] @ c
            #          V[:,elim] = V[:,other] @ c  
            #          W[:,elim] = W[:,other] @ c
            # where c is the same for all three
            
            # Build combined system
            U_mat = np.column_stack([U_arr[:, j] for j in other_idx])
            V_mat = np.column_stack([V_arr[:, j] for j in other_idx])
            W_mat = np.column_stack([W_arr[:, j] for j in other_idx])
            
            U_tgt = U_arr[:, elim]
            V_tgt = V_arr[:, elim]
            W_tgt = W_arr[:, elim]
            
            A = np.vstack([U_mat, V_mat, W_mat])
            b = np.concatenate([U_tgt, V_tgt, W_tgt])
            
            try:
                # Solve least squares
                c, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if exact
                recon = A @ c
                if np.allclose(recon, b, atol=1e-9):
                    # Found it! Create new decomposition
                    new_U = np.array(U, dtype=float)
                    new_V = np.array(V, dtype=float)
                    new_W = np.array(W, dtype=float)
                    
                    # Subtract the linear combination
                    for idx, j in enumerate(other_idx):
                        if abs(c[idx]) > 1e-12:
                            new_U[:, j] += c[idx] * U_arr[:, elim]
                            new_V[:, j] += c[idx] * V_arr[:, elim]
                            new_W[:, j] += c[idx] * W_arr[:, elim]
                    
                    # Remove the eliminated column
                    mask = [j for j in range(current_rank) if j != elim]
                    new_U = new_U[:, mask]
                    new_V = new_V[:, mask]
                    new_W = new_W[:, mask]
                    
                    # Convert back to list of lists
                    return (new_U.tolist(), new_V.tolist(), new_W.tolist(), current_rank - 1)
            except:
                pass
        
        return None
    
    # Build initial Strassen decomposition
    U, V, W, rank = build_strassen()
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback: return schoolbook rank 32
        U = [[0.0] * 32 for _ in range(n * m)]
        V = [[0.0] * 32 for _ in range(m * p)]
        W = [[0.0] * 32 for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank by peeling
    max_attempts = 5
    for attempt in range(max_attempts):
        if time_up():
            break
        
        result = try_peel(U, V, W, rank)
        if result is None:
            break
        
        U, V, W, rank = result
        
        if verify(U, V, W, rank):
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
