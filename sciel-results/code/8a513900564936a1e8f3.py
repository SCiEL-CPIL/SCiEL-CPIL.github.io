import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 3, 4, 4
    
    def build_rank44():
        """Build rank-44 decomposition using 4 blocks of 3x2x2 at rank 11."""
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Local 3x2x2 factor with rank 11
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen for rows 0,1 (cols 0-6)
        # M1: (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        # M2: (a10+a11)b00 -> c10+=M2, c11-=M2
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        # M3: a00(b11-b01) -> c01+=M3, c11+=M3
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        # M4: a11(b10-b00) -> c00+=M4, c10+=M4
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        # M5: (a00+a01)b11 -> c00-=M5, c01+=M5
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        # M6: (a10-a00)(b00+b01) -> c11+=M6
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        # M7: (a01-a11)(b10+b11) -> c00+=M7
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Trivial for row 2 (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    la = r * 2 + c
                    ga = r * m + (a_col_start + c)
                    U[ga, col_offset:col_offset+11] += U_loc[la, :]
            for r in range(2):
                for c in range(2):
                    lb = r * 2 + c
                    gb = (b_row_start + r) * p + (b_col_start + c)
                    V[gb, col_offset:col_offset+11] += V_loc[lb, :]
            for r in range(3):
                for c in range(2):
                    lc = r * 2 + c
                    gc = (c_col_start + c) * n + r
                    W[gc, col_offset:col_offset+11] += W_loc[lc, :]
        
        add_block(0, 0, 0, 0, 0)
        add_block(1, 2, 2, 0, 0)
        add_block(2, 0, 0, 2, 2)
        add_block(3, 2, 2, 2, 2)
        
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                for c in range(p*n):
                    c_col, c_row = divmod(c, n)
                    expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
                    computed = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(computed - expected) > 1e-10:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, target_rank, max_attempts=500):
        """Try to reduce rank by finding redundant columns."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        current_rank = U.shape[1]
        
        if current_rank <= target_rank:
            return U[:, :current_rank], V[:, :current_rank], W[:, :current_rank]
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        rng = np.random.RandomState(42)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            if best_rank <= target_rank:
                break
            
            # Try to eliminate a random column
            col = rng.randint(0, best_rank)
            
            # Check if this column can be expressed as a combination of others
            # by checking if removing it and adjusting others preserves the tensor
            # This is hard, so use a heuristic: check if the column has low "importance"
            
            # Compute contribution of each column to the tensor
            # For column r, its contribution to T[a,b,c] is U[a,r]*V[b,r]*W[c,r]
            
            # Try: zero out column, see if we can fix with small adjustments
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            U_test[:, col] = 0
            V_test[:, col] = 0
            W_test[:, col] = 0
            
            # Check if still valid (it won't be, but this is a starting point)
            # Instead, try to find a linear dependency
            
            # Alternative: try merging two columns
            col2 = rng.randint(0, best_rank)
            if col2 == col:
                continue
            
            # Try: new_col = alpha * col + beta * col2
            # This is underdetermined, so just try simple cases
            
            # For now, just try to see if any column is exactly redundant
            # by checking if U[:,col] is in span of other columns (approximately)
            other_cols = list(range(best_rank))
            other_cols.remove(col)
            
            if len(other_cols) < 1:
                continue
            
            # Check rank of U without this column vs with
            U_without = best_U[:, other_cols]
            U_with = best_U[:, :best_rank]
            
            if np.linalg.matrix_rank(U_with) == np.linalg.matrix_rank(U_without):
                # Column might be redundant in U, check V and W too
                V_without = best_V[:, other_cols]
                V_with = best_V[:, :best_rank]
                W_without = best_W[:, other_cols]
                W_with = best_W[:, :best_rank]
                
                if np.linalg.matrix_rank(V_with) == np.linalg.matrix_rank(V_without) and \
                   np.linalg.matrix_rank(W_with) == np.linalg.matrix_rank(W_without):
                    # Try removing this column
                    mask = [i for i in range(best_rank) if i != col]
                    U_new = best_U[:, mask]
                    V_new = best_V[:, mask]
                    W_new = best_W[:, mask]
                    
                    # Verify
                    if verify(U_new, V_new, W_new):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank -= 1
        
        return best_U[:, :best_rank], best_V[:, :best_rank], best_W[:, :best_rank]
    
    # Build initial rank-44 decomposition
    U, V, W = build_rank44()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school))
        V_fb = np.zeros((m*p, rank_school))
        W_fb = np.zeros((p*n, rank_school))
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col == b_row:
                    c_idx = b_col * n + a_row
                    rank_idx = a * (m*p) + b
                    U_fb[a, rank_idx] = 1
                    V_fb[b, rank_idx] = 1
                    W_fb[c_idx, rank_idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, 42)
    
    # Verify reduced version
    if verify(U_reduced, V_reduced, W_reduced):
        U, V, W = U_reduced, V_reduced, W_reduced
    
    # Final verification
    if not verify(U, V, W):
        # Use the original rank-44
        U, V, W = build_rank44()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
