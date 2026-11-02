import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Efficient verification using tensor contraction
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We need T[a,b,c] = 1 iff a_col==b_row and c_row==a_row and c_col==b_col
        
        # Build expected tensor
        expected = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col == b_row:
                    c_idx = b_col * n + a_row
                    expected[a, b, c_idx] = 1.0
        
        # Compute actual tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use broadcasting: U[a,r]*V[b,r] gives (n*m, m*p, rank), then * W[c,r] and sum
        # This is memory intensive for large rank, so do it in chunks
        chunk_size = max(1, (5000000 // (rank * (n*m))))  # ~5M elements per chunk
        for c_start in range(0, p*n, chunk_size):
            c_end = min(c_start + chunk_size, p*n)
            # W_chunk: (c_end-c_start, rank)
            W_chunk = W[c_start:c_end, :]
            # Compute sum over r: U[a,r] * V[b,r] * W_chunk[c,r]
            # = sum_r (U[a,r]*W_chunk[c,r]) * V[b,r]
            # Let's do: for each c, sum_r U[a,r]*V[b,r]*W[c,r]
            # = (U @ diag(V) @ W.T) ... no
            # Actually: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # = sum_r (U[:,r] * V[:,r]) * W[c,r]
            # Let M = U * V.T  (n*m, rank) * (rank, m*p) -> no
            # M[a,b] = sum_r U[a,r]*V[b,r]  -> this is U @ V.T  (n*m, m*p)
            # Then T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # This is NOT just U@V.T * W[c,r] because the r is shared
            # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # For fixed c: T[:,:,c] = sum_r U[:,r]*V[:,r]*W[c,r]
            # = sum_r W[c,r] * (U[:,r] outer V[:,r])
            # This is a rank-1 update per r, weighted by W[c,r]
            # Efficient: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r].T)
            # = (U * W[c,:][None,:]) @ V.T  ... let's check
            # (U * W[c,:][None,:])[a,r] = U[a,r]*W[c,r]
            # ((U * W[c,:][None,:]) @ V.T)[a,b] = sum_r U[a,r]*W[c,r]*V[b,r]  YES!
            
            for c_idx in range(c_start, c_end):
                # T[:,:,c_idx] = (U * W[c_idx,:][None,:]) @ V.T
                actual = (U * W[c_idx, :][None, :]) @ V.T
                if not np.allclose(actual, expected[:, :, c_idx], atol=1e-10):
                    return False
        return True
    
    def build_rank44():
        """Build rank-44 decomposition using 4 blocks of 3x2x2 at rank 11."""
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Local 3x2x2 factor with rank 11
        # Strassen for 2x2x2 gives rank 7, plus 4 trivial for the 3rd row = 11
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
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[5, 8] = 1
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
    
    def try_reduce_rank(U, V, W, target_rank, max_attempts=200):
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
            if time.time() > deadline - 15:
                break
            if best_rank <= target_rank:
                break
            
            # Try to eliminate a random column
            col = rng.randint(0, best_rank)
            
            # Check if this column is linearly dependent in U, V, W simultaneously
            other_cols = [i for i in range(best_rank) if i != col]
            
            # Check rank of U without this column
            U_without = best_U[:, other_cols]
            U_with = best_U[:, :best_rank]
            
            # Use SVD to check rank
            rank_U_with = np.linalg.matrix_rank(U_with)
            rank_U_without = np.linalg.matrix_rank(U_without)
            
            if rank_U_with > rank_U_without:
                continue  # Column is essential in U
            
            rank_V_with = np.linalg.matrix_rank(best_V[:, :best_rank])
            rank_V_without = np.linalg.matrix_rank(best_V[:, other_cols])
            
            if rank_V_with > rank_V_without:
                continue  # Column is essential in V
            
            rank_W_with = np.linalg.matrix_rank(best_W[:, :best_rank])
            rank_W_without = np.linalg.matrix_rank(best_W[:, other_cols])
            
            if rank_W_with > rank_W_without:
                continue  # Column is essential in W
            
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
    
    def try_pairwise_merge(U, V, W, max_attempts=100):
        """Try to merge pairs of columns to reduce rank by 1."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        current_rank = U.shape[1]
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        rng = np.random.RandomState(123)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Try to merge two random columns
            cols = rng.choice(best_rank, size=2, replace=False)
            i, j = int(cols[0]), int(cols[1])
            
            if i == j:
                continue
            
            # Try: replace column i with (i + j) and remove column j
            # This works if the tensor contribution of j can be absorbed
            # Actually, we need: sum_r U[a,r]V[b,r]W[c,r] unchanged
            # If we set new_i = i + alpha*j, we need to find alpha such that
            # the tensor is preserved. This is generally not possible for arbitrary alpha.
            
            # Alternative: try all simple integer combinations
            # new_i = i + j, remove j
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            U_test[:, i] = best_U[:, i] + best_U[:, j]
            V_test[:, i] = best_V[:, i] + best_V[:, j]
            W_test[:, i] = best_W[:, i] + best_W[:, j]
            
            # Remove column j
            mask = [k for k in range(best_rank) if k != j]
            U_test = U_test[:, mask]
            V_test = V_test[:, mask]
            W_test = W_test[:, mask]
            
            if verify(U_test, V_test, W_test):
                best_U, best_V, best_W = U_test, V_test, W_test
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
    
    # Try to reduce rank by removing redundant columns
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, 42)
    
    # Verify reduced version
    if verify(U_reduced, V_reduced, W_reduced):
        U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try pairwise merge
    U_merged, V_merged, W_merged = try_pairwise_merge(U, V, W, 50)
    
    # Verify merged version
    if verify(U_merged, V_merged, W_merged):
        U, V, W = U_merged, V_merged, W_merged
    
    # Final verification
    if not verify(U, V, W):
        # Use the original rank-44
        U, V, W = build_rank44()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
