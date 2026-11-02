"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build the Strassen-based rank-28 decomposition for 2x4 * 4x8
    # This uses two 2x2*2x2 Strassen blocks for the left half and right half
    def build_strassen_28():
        U_s = np.zeros((U_dim, 28))
        V_s = np.zeros((V_dim, 28))
        W_s = np.zeros((W_dim, 28))
        
        # For 2x2 * 2x2 Strassen, we have 7 multiplications:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        # M5 = (A00+A10)B10 -> C00 -= M5, C01 += M5
        # M6 = (A01-A10)(B10+B11) -> C11 += M6
        # M7 = (A10-A11)(B00+B01) -> C00 += M7
        
        def add_strassen_block(base_col, A_rows, B_cols, C_rows):
            """
            Add a Strassen block.
            A_rows: [row0, row1] in U space (indices i*m+j)
            B_cols: [col0, col1] in V space (indices j*p+k)
            C_rows: [row0, row1] in W space (indices k*n+i)
            
            For the 2x4*4x8 case:
            U has shape (2,4) -> indices i*4+j for i in {0,1}, j in {0,1,2,3}
            V has shape (4,8) -> indices j*8+k for j in {0,1,2,3}, k in {0,1,2,3}
            W has shape (8,2) -> indices k*2+i for k in {0,1,2,3}, i in {0,1}
            
            We split into left (cols 0-1 of B) and right (cols 2-3 of B)
            """
            # A00, A01, A10, A11 in U space
            A00 = A_rows[0] * 4 + 0
            A01 = A_rows[0] * 4 + 1
            A10 = A_rows[1] * 4 + 0
            A11 = A_rows[1] * 4 + 1
            
            # B00, B01, B10, B11 in V space
            # B is 4x8, we take columns B_cols[0] and B_cols[1]
            # B00 = row 0, col B_cols[0]
            # B01 = row 0, col B_cols[1]
            # B10 = row 1, col B_cols[0]
            # B11 = row 1, col B_cols[1]
            B00 = 0 * 8 + B_cols[0]
            B01 = 0 * 8 + B_cols[1]
            B10 = 1 * 8 + B_cols[0]
            B11 = 1 * 8 + B_cols[1]
            
            # C00, C01, C10, C11 in W space
            # C is 8x2, we take rows C_rows[0] and C_rows[1]
            # C00 = row C_rows[0], col 0
            # C01 = row C_rows[0], col 1
            # C10 = row C_rows[1], col 0
            # C11 = row C_rows[1], col 1
            C00 = C_rows[0] * 2 + 0
            C01 = C_rows[0] * 2 + 1
            C10 = C_rows[1] * 2 + 0
            C11 = C_rows[1] * 2 + 1
            
            # M1 = (A00+A11)(B00+B11)
            col = base_col
            U_s[A00, col] = 1
            U_s[A11, col] = 1
            V_s[B00, col] = 1
            V_s[B11, col] = 1
            W_s[C00, col] = 1
            W_s[C11, col] = 1
            
            # M2 = (A10+A11)B00
            col = base_col + 1
            U_s[A10, col] = 1
            U_s[A11, col] = 1
            V_s[B00, col] = 1
            W_s[C10, col] = 1
            W_s[C11, col] = -1
            
            # M3 = A00(B01-B11)
            col = base_col + 2
            U_s[A00, col] = 1
            V_s[B01, col] = 1
            V_s[B11, col] = -1
            W_s[C01, col] = 1
            W_s[C11, col] = 1
            
            # M4 = A11(B10-B00)
            col = base_col + 3
            U_s[A11, col] = 1
            V_s[B10, col] = 1
            V_s[B00, col] = -1
            W_s[C00, col] = 1
            W_s[C10, col] = 1
            
            # M5 = (A00+A10)B10
            col = base_col + 4
            U_s[A00, col] = 1
            U_s[A10, col] = 1
            V_s[B10, col] = 1
            W_s[C00, col] = -1
            W_s[C01, col] = 1
            
            # M6 = (A01-A10)(B10+B11)
            col = base_col + 5
            U_s[A01, col] = 1
            U_s[A10, col] = -1
            V_s[B10, col] = 1
            V_s[B11, col] = 1
            W_s[C11, col] = 1
            
            # M7 = (A10-A11)(B00+B01)
            col = base_col + 6
            U_s[A10, col] = 1
            U_s[A11, col] = -1
            V_s[B00, col] = 1
            V_s[B01, col] = 1
            W_s[C00, col] = 1
            
            return base_col + 7
        
        # For 2x4 * 4x8:
        # A is 2x4, B is 4x8, C is 8x2
        # We can think of this as:
        # C[0:2, :] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]  -- wait, this isn't right
        # Actually, C[i,k] = sum_j A[i,j] * B[j,k]
        # C has 8 rows (k=0..7) and 2 cols (i=0,1)
        # A has 2 rows (i=0,1) and 4 cols (j=0..3)
        # B has 4 rows (j=0..3) and 8 cols (k=0..7)
        
        # So C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        # We can split this into two 2x2*2x8 multiplications:
        # Left: C[:, 0:4] = A[:, 0:2] * B[0:2, 0:4]
        # Right: C[:, 4:8] = A[:, 2:4] * B[2:4, 0:4]
        
        # But each of these is 2x2 * 2x4, which can be done with Strassen-like methods.
        # Actually, let's think of it as two separate 2x2 * 2x4 problems.
        
        # For 2x2 * 2x4:
        # A is 2x2, B is 2x4, C is 4x2
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # We can split B into two 2x2 blocks: B_left = B[:,0:2], B_right = B[:,2:4]
        # C[:,0:2] = A * B_left (2x2*2x2)
        # C[:,2:4] = A * B_right (2x2*2x2)
        
        # So we need 4 Strassen blocks of 2x2*2x2:
        # 1. A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # 2. A[:,0:2] * B[0:2,2:4] -> C[2:4,0:2]
        # 3. A[:,2:4] * B[2:4,0:2] -> C[0:2,2:4]
        # 4. A[:,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        
        # Wait, I need to be more careful with the indexing.
        
        # Let me redefine:
        # U has shape (8, rank) where index i*4+j corresponds to A[i,j]
        # V has shape (32, rank) where index j*8+k corresponds to B[j,k]
        # W has shape (16, rank) where index k*2+i corresponds to C[k,i]
        
        # For the 4 Strassen blocks:
        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        #   A_rows = [0,1], B_cols = [0,1], C_rows = [0,1]
        # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
        #   A_rows = [0,1], B_cols = [2,3], C_rows = [2,3]
        # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4]
        #   A_rows = [0,1], B_cols = [0,1], C_rows = [0,1] but with A cols 2,3
        # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        #   A_rows = [0,1], B_cols = [2,3], C_rows = [2,3] but with A cols 2,3
        
        # I need to modify add_strassen_block to handle the A column offset.
        
        def add_strassen_block_v2(base_col, A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset):
            """
            Add a Strassen block with offsets.
            A[i,j] is at U index (A_row_offset+i)*4 + (A_col_offset+j)
            B[i,j] is at V index (B_row_offset+i)*8 + (B_col_offset+j)
            C[i,j] is at W index (C_row_offset+i)*2 + j
            """
            # A00, A01, A10, A11 in U space
            A00 = (A_row_offset + 0) * 4 + (A_col_offset + 0)
            A01 = (A_row_offset + 0) * 4 + (A_col_offset + 1)
            A10 = (A_row_offset + 1) * 4 + (A_col_offset + 0)
            A11 = (A_row_offset + 1) * 4 + (A_col_offset + 1)
            
            # B00, B01, B10, B11 in V space
            B00 = (B_row_offset + 0) * 8 + (B_col_offset + 0)
            B01 = (B_row_offset + 0) * 8 + (B_col_offset + 1)
            B10 = (B_row_offset + 1) * 8 + (B_col_offset + 0)
            B11 = (B_row_offset + 1) * 8 + (B_col_offset + 1)
            
            # C00, C01, C10, C11 in W space
            C00 = (C_row_offset + 0) * 2 + 0
            C01 = (C_row_offset + 0) * 2 + 1
            C10 = (C_row_offset + 1) * 2 + 0
            C11 = (C_row_offset + 1) * 2 + 1
            
            # M1 = (A00+A11)(B00+B11)
            col = base_col
            U_s[A00, col] = 1
            U_s[A11, col] = 1
            V_s[B00, col] = 1
            V_s[B11, col] = 1
            W_s[C00, col] = 1
            W_s[C11, col] = 1
            
            # M2 = (A10+A11)B00
            col = base_col + 1
            U_s[A10, col] = 1
            U_s[A11, col] = 1
            V_s[B00, col] = 1
            W_s[C10, col] = 1
            W_s[C11, col] = -1
            
            # M3 = A00(B01-B11)
            col = base_col + 2
            U_s[A00, col] = 1
            V_s[B01, col] = 1
            V_s[B11, col] = -1
            W_s[C01, col] = 1
            W_s[C11, col] = 1
            
            # M4 = A11(B10-B00)
            col = base_col + 3
            U_s[A11, col] = 1
            V_s[B10, col] = 1
            V_s[B00, col] = -1
            W_s[C00, col] = 1
            W_s[C10, col] = 1
            
            # M5 = (A00+A10)B10
            col = base_col + 4
            U_s[A00, col] = 1
            U_s[A10, col] = 1
            V_s[B10, col] = 1
            W_s[C00, col] = -1
            W_s[C01, col] = 1
            
            # M6 = (A01-A10)(B10+B11)
            col = base_col + 5
            U_s[A01, col] = 1
            U_s[A10, col] = -1
            V_s[B10, col] = 1
            V_s[B11, col] = 1
            W_s[C11, col] = 1
            
            # M7 = (A10-A11)(B00+B01)
            col = base_col + 6
            U_s[A10, col] = 1
            U_s[A11, col] = -1
            V_s[B00, col] = 1
            V_s[B01, col] = 1
            W_s[C00, col] = 1
            
            return base_col + 7
        
        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        next_col = add_strassen_block_v2(0, 0, 0, 0, 0, 0)
        
        # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
        next_col = add_strassen_block_v2(next_col, 0, 0, 0, 2, 2)
        
        # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4]
        next_col = add_strassen_block_v2(next_col, 0, 2, 2, 0, 0)
        
        # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        next_col = add_strassen_block_v2(next_col, 0, 2, 2, 2, 2)
        
        return U_s, V_s, W_s
    
    U_str, V_str, W_str = build_strassen_28()
    
    if check_decomp(U_str, V_str, W_str):
        U, V, W = U_str, V_str, W_str
        current_rank = 28
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        current_rank = 64
    
    # Now try to reduce rank using a different approach:
    # Use the fact that we can combine some of the Strassen terms
    # by finding linear dependencies.
    
    def try_reduce_by_combination(U, V, W, max_iter=100):
        """
        Try to reduce rank by finding columns that are linear combinations
        of others and eliminating them.
        """
        rank = U.shape[1]
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Try to find a column that can be eliminated
            # A column r can be eliminated if there exist coefficients c_s (s != r)
            # such that for all (i,j,k):
            # sum_s c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 
            # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            
            # This is a nonlinear condition, but we can try a simpler approach:
            # Check if the column r is a linear combination of other columns
            # in the "tensor product" sense.
            
            # For each column r, try to express it as a combination of others
            # by solving a least squares problem.
            
            best_elimination = None
            best_residual = float('inf')
            
            for r in range(rank):
                if rank <= 16:  # Don't go below 16 (theoretical lower bound)
                    break
                
                # Try to eliminate column r
                mask = np.ones(rank, dtype=bool)
                mask[r] = False
                
                # Build the constraint matrix
                # For each (i,j,k), we need:
                # sum_s c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 
                # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                
                all_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                num_constraints = min(len(all_indices), 50)
                sampled = random.sample(all_indices, num_constraints)
                
                A_mat = np.zeros((num_constraints, rank - 1))
                b_vec = np.zeros(num_constraints)
                
                for idx, (i, j, k) in enumerate(sampled):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    for s, col_idx in enumerate(np.where(mask)[0]):
                        A_mat[idx, s] = U_curr[u_idx, col_idx] * V_curr[v_idx, col_idx] * W_curr[w_idx, col_idx]
                    
                    b_vec[idx] = U_curr[u_idx, r] * V_curr[v_idx, r] * W_curr[w_idx, r]
                
                try:
                    c_sol, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    residual = np.linalg.norm(A_mat @ c_sol - b_vec)
                    
                    if residual < 1e-6 and residual < best_residual:
                        best_residual = residual
                        best_elimination = (r, c_sol, np.where(mask)[0])
                except:
                    pass
            
            if best_elimination is not None:
                r, c_sol, cols = best_elimination
                
                # Eliminate column r by adding c_s * column_s to column r
                # Actually, we want to set column r to zero and adjust other columns
                # But this is tricky. Let's just remove column r and see if it still works.
                
                U_test = np.delete(U_curr, r, axis=1)
                V_test = np.delete(V_curr, r, axis=1)
                W_test = np.delete(W_curr, r, axis=1)
                
                if check_decomp(U_test, V_test, W_test, tol=1e-5):
                    U_curr, V_curr, W_curr = U_test, V_test, W_test
                    rank -= 1
            
            if rank <= 16:
                break
        
        return U_curr, V_curr, W_curr
    
    # Try to reduce the rank
    U_new, V_new, W_new = try_reduce_by_combination(U, V, W)
    
    if check_decomp(U_new, V_new, W_new) and U_new.shape[1] < current_rank:
        U, V, W = U_new, V_new, W_new
        current_rank = U_new.shape[1]
    
    # Final verification
    if not check_decomp(U, V, W):
        # Fall back to Strassen if valid, else schoolbook
        if check_decomp(U_str, V_str, W_str):
            U, V, W = U_str, V_str, W_str
        else:
            rank = n * m * p
            U = np.zeros((U_dim, rank))
            V = np.zeros((V_dim, rank))
            W = np.zeros((W_dim, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
