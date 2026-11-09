"""Tensor decomposition search for 2x5 x 5x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

def construct():
    """Search for rank-minimal tensor decomposition."""
    deadline = time.time() + 140  # 140s budget, leaving 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat):
        """Check if decomposition is exact."""
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U_mat.shape[1]
        # Compute tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Try to reduce rank by removing columns and solving linear system
    def try_reduce(U_cur, V_cur, W_cur, current_rank, max_attempts=200):
        """Try to reduce rank by 1 using linearization."""
        n, m, p = DIMENSIONS
        if time.time() > deadline - 5:
            return None, None, None, current_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None, None, None, current_rank
            
            # Randomly choose which column to remove
            remove_col = random.randint(0, current_rank - 1)
            
            # Build the "hole" tensor: what was contributed by column remove_col
            hole = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        hole[i, j, k] = U_cur[i * m + j, remove_col] * V_cur[j * p + k, remove_col] * W_cur[k * n + i, remove_col]
            
            # Check if hole is zero (nothing to compensate)
            if np.allclose(hole, 0, atol=1e-12):
                # Just remove the column
                mask = [c for c in range(current_rank) if c != remove_col]
                U_new = U_cur[:, mask]
                V_new = V_cur[:, mask]
                W_new = W_cur[:, mask]
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new, current_rank - 1
                continue
            
            # Build linear system: we need to find deltas such that
            # sum_{r != remove_col} [U_r dV_r W_r + U_r V_r dW_r + dU_r V_r W_r] = hole
            # Variables: dU[i*m+j, r], dV[j*p+k, r], dW[k*n+i, r] for all r != remove_col
            # Number of variables: 3 * current_rank * ... wait, let me think again
            
            # Actually, the perturbation for each remaining column r is:
            # delta_r = dU_r * V_r * W_r + U_r * dV_r * W_r + U_r * V_r * dW_r
            # This is linear in (dU_r, dV_r, dW_r) for fixed U_r, V_r, W_r
            
            # Let's set up: for each remaining column r, we have 3 scalar variables:
            # a_r (multiplies U_r*V_r*W_r contribution via dU), etc.
            # Actually it's simpler: define for each r, variables x_r, y_r, z_r
            # where the contribution is x_r * (U_r V_r W_r component) ... 
            # 
            # Let me reconsider. The product at position (i,j,k) for column r is:
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # If we perturb U[i*m+j, r] by dU, the change is dU * V[j*p+k, r] * W[k*n+i, r]
            # Similarly for V and W.
            #
            # So for each remaining column r and each (i,j,k), the contribution to the correction is:
            # dU[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
            # + U[i*m+j, r] * dV[j*p+k, r] * W[k*n+i, r]
            # + U[i*m+j, r] * V[j*p+k, r] * dW[k*n+i, r]
            #
            # This is linear in the dU, dV, dW variables.
            #
            # Total variables: 3 * current_rank (since we have current_rank-1 remaining columns, but each has 3 sets of values)
            # Actually, dU has shape (n*m, current_rank-1), dV has (m*p, current_rank-1), dW has (p*n, current_rank-1)
            # Total: (n*m + m*p + p*n) * (current_rank - 1)
            
            nm, mp, pn = n * m, m * p, p * n
            remaining = [r for r in range(current_rank) if r != remove_col]
            num_remaining = len(remaining)
            num_vars = (nm + mp + pn) * num_remaining
            num_eqs = n * m * p  # 50 equations
            
            # Build A matrix (num_eqs x num_vars) and b vector (num_eqs,)
            A = np.zeros((num_eqs, num_vars))
            b = hole.flatten().copy()
            
            # Map (i,j,k) to equation index
            def eq_idx(i, j, k):
                return i * m * p + j * p + k
            
            # For each remaining column r_idx (0..num_remaining-1), actual column index is remaining[r_idx]
            for r_idx, r in enumerate(remaining):
                # dU variables: indices r_idx * (nm + mp + pn) to r_idx * (nm + mp + pn) + nm - 1
                base_u = r_idx * (nm + mp + pn)
                base_v = base_u + nm
                base_w = base_v + mp
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ei = eq_idx(i, j, k)
                            Ui = U_cur[i * m + j, r]
                            Vj = V_cur[j * p + k, r]
                            Wk = W_cur[k * n + i, r]
                            
                            # dU contribution
                            if Vj != 0 and Wk != 0:
                                A[ei, base_u + i * m + j] = Vj * Wk
                            # dV contribution
                            if Ui != 0 and Wk != 0:
                                A[ei, base_v + j * p + k] = Ui * Wk
                            # dW contribution
                            if Ui != 0 and Vj != 0:
                                A[ei, base_w + k * n + i] = Ui * Vj
            
            # Solve least squares
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Extract deltas
            U_new = U_cur.copy()
            V_new = V_cur.copy()
            W_new = W_cur.copy()
            
            for r_idx, r in enumerate(remaining):
                base_u = r_idx * (nm + mp + pn)
                base_v = base_u + nm
                base_w = base_v + mp
                
                dU = x[base_u:base_u + nm].reshape(n, m)
                dV = x[base_v:base_v + mp].reshape(m, p)
                dW = x[base_w:base_w + pn].reshape(p, n)
                
                # Round to half-integers
                dU_rounded = np.round(dU * 2) / 2
                dV_rounded = np.round(dV * 2) / 2
                dW_rounded = np.round(dW * 2) / 2
                
                # Apply to column r
                U_new[:, r] += dU_rounded.flatten()
                V_new[:, r] += dV_rounded.flatten()
                W_new[:, r] += dW_rounded.flatten()
            
            # Remove the target column
            mask = [c for c in range(current_rank) if c != remove_col]
            U_final = U_new[:, mask]
            V_final = V_new[:, mask]
            W_final = W_new[:, mask]
            
            if verify(U_final, V_final, W_final):
                return U_final, V_final, W_final, current_rank - 1
        
        return None, None, None, current_rank
    
    # Iteratively try to reduce rank
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    current_rank = rank_schoolbook
    
    for target_rank in range(rank_schoolbook - 1, max(20, rank_schoolbook - 10), -1):
        if time.time() > deadline - 10:
            break
        
        U_new, V_new, W_new, new_rank = try_reduce(U_cur, V_cur, W_cur, current_rank, max_attempts=150)
        if U_new is not None and new_rank < current_rank:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            current_rank = new_rank
            if current_rank < best_rank:
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                best_rank = current_rank
    
    # Return best found
    if best_rank < rank_schoolbook:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U),
            "V": to_list(V),
            "W": to_list(W)
        }
