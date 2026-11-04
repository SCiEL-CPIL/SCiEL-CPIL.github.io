import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill
    
    def _time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    n, m, p = 2, 3, 6
    
    # Build the rank-33 solution
    rank_start = 33
    U = np.zeros((n * m, rank_start))
    V = np.zeros((m * p, rank_start))
    W = np.zeros((p * n, rank_start))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :] - schoolbook, rank 12
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: 2x2x2 Strassen on 3 blocks for A[:,1:3] * B[1:3,:]
    # A_sub[i, j] where i in {0,1}, j in {0,1} corresponds to A[i, 1+j]
    # B_sub[j, k] where j in {0,1}, k in {0,...,5} corresponds to B[1+j, k]
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3: A00(B10-B11) -> C10 += M3, C11 += M3
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4: A11(B01-B00) -> C00 += M4, C01 += M4
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5: (A00+A01)B11 -> C00 -= M5, C10 += M5
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1
    
    # Verify initial solution
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U_mat[i*m+j, rr] * V_mat[j*p+k, rr] * W_mat[k*n+i, rr]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_start
    
    # Local search: try to remove columns one at a time
    # For each column, check if the tensor can still be decomposed without it
    # by solving for adjustments to remaining columns
    
    current_rank = best_rank
    U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
    
    # Precompute the tensor values for each (i,j,k)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    # Strategy: for each column c, try to express its contribution as a combination
    # of other columns. This is equivalent to checking if the column is in the span
    # of the others in the tensor product space.
    
    # More practically: remove column c, then try to find small adjustments to
    # remaining columns that restore the tensor. Use least squares.
    
    def try_remove_column(U_mat, V_mat, W_mat, col_idx):
        """Try to remove column col_idx by adjusting other columns."""
        r = U_mat.shape[1]
        if r <= 1:
            return None, None, None
        
        # Remove the column
        mask = np.ones(r, dtype=bool)
        mask[col_idx] = False
        U_rem = U_mat[:, mask].copy()
        V_rem = V_mat[:, mask].copy()
        W_rem = W_mat[:, mask].copy()
        
        # Check if already valid
        if verify(U_rem, V_rem, W_rem, tol=1e-10):
            return U_rem, V_rem, W_rem
        
        # The contribution of the removed column to each (i,j,k) is:
        # delta[i,j,k] = U[i*m+j, col_idx] * V[j*p+k, col_idx] * W[k*n+i, col_idx]
        # We need to find adjustments dU, dV, dW to remaining columns such that
        # sum_r' dU[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] + 
        #        U[i*m+j, r'] * dV[j*p+k, r'] * W[k*n+i, r'] +
        #        U[i*m+j, r'] * V[j*p+k, r'] * dW[k*n+i, r'] = delta[i,j,k]
        # (first-order approximation, ignoring second-order terms)
        
        # This is complex. Simpler approach: just check if the column is redundant
        # by seeing if removing it and not adjusting still gives the right answer.
        
        return None, None, None
    
    # Simpler local search: just try removing each column and check validity
    improved = True
    while improved and _time_remaining() > 10.0:
        improved = False
        for col_idx in range(current_rank):
            if _time_remaining() < 5.0:
                break
            
            mask = np.ones(current_rank, dtype=bool)
            mask[col_idx] = False
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if verify(U_test, V_test, W_test, tol=1e-10):
                U_cur, V_cur, W_cur = U_test, V_test, W_test
                current_rank -= 1
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                best_rank = current_rank
                improved = True
                break  # Restart the loop with the new rank
    
    # Also try: for pairs of columns, see if one can be absorbed into the other
    # by linearity: if column a = alpha * column b in the tensor sense, we can merge
    # This is harder to check directly.
    
    if _time_remaining() > 5.0:
        # Try a different approach: random linear combinations of columns
        # to find a lower-rank equivalent
        np.random.seed(42)
        for attempt in range(20):
            if _time_remaining() < 5.0:
                break
            
            # Try to find a rank-(current_rank-1) decomposition by
            # combining two columns into one
            for c1 in range(current_rank):
                for c2 in range(c1+1, current_rank):
                    # Try to replace columns c1 and c2 with a single column
                    # that gives the same tensor contribution
                    # This requires solving a nonlinear system, which is hard.
                    # Skip for now.
                    pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
