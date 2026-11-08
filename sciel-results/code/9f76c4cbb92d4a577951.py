import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Helper to solve for a block M_{n, m, p_block}
    def solve_block(n, m, p_block, target_rank):
        # U: (n*m, R), V: (m*p_block, R), W: (p_block*n, R)
        # Equations: sum_r U[i*m+j, r] V[j*p_block+k, r] W[k*n+i, r] = 1 for all i,j,k
        num_eqs = n * m * p_block
        
        def residuals(params):
            R = target_rank
            U = params[:n*m*R].reshape(n*m, R)
            V = params[n*m*R:n*m*R + m*p_block*R].reshape(m*p_block, R)
            W = params[n*m*R + m*p_block*R:].reshape(p_block*n, R)
            
            res = np.zeros(num_eqs)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p_block):
                        u_val = U[i*m + j]
                        v_val = V[j*p_block + k]
                        w_val = W[k*n + i]
                        # sum_r u_v_w
                        val = np.dot(u_val, v_val * w_val)
                        res[idx] = val - 1.0
                        idx += 1
            return res

        best_params = None
        best_err = np.inf
        
        # Try a few random starts
        num_starts = 20
        for _ in range(num_starts):
            if time.time() > deadline:
                break
            R = target_rank
            total_vars = (n*m + m*p_block + p_block*n) * R
            # Initialize with small random values
            params = np.random.randn(total_vars) * 0.1
            try:
                sol = least_squares(residuals, params, method='lm', max_nfev=1000)
                if sol.cost < best_err:
                    best_err = sol.cost
                    best_params = sol.x
            except Exception:
                pass
        
        if best_err < 1e-6:
            R = target_rank
            U = best_params[:n*m*R].reshape(n*m, R)
            V = best_params[n*m*R:n*m*R + m*p_block*R].reshape(m*p_block, R)
            W = best_params[n*m*R + m*p_block*R:].reshape(p_block*n, R)
            # Round to nearest integer or half-integer?
            # The problem allows integer/half-integer.
            # If the solution is close to integers, we can snap.
            # But least_squares might give floats.
            # We verify the rounded version.
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            # Check error of rounded
            def check_error(U, V, W):
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p_block):
                            val = np.dot(U[i*m+j], V[j*p_block+k] * W[k*n+i])
                            err += (val - 1.0)**2
                return err
                
            if check_error(U_rounded, V_rounded, W_rounded) < 1e-9:
                return U_rounded, V_rounded, W_rounded
            else:
                # If rounding failed, maybe the rank is not this low or we need more precision
                # Return None to indicate failure
                return None
        else:
            return None

    # Strategy: Split p=8 into 3, 3, 2
    # Block 1: 2x3x3, Target Rank 11
    # Block 2: 2x3x3, Target Rank 11
    # Block 3: 2x3x2, Target Rank 10 or 11
    
    blocks = []
    
    # Try to solve Block 1 (2x3x3) with Rank 11
    sol1 = solve_block(n, m, 3, 11)
    if sol1 is None:
        # Fallback to schoolbook for this block (Rank 9)
        # But we want to minimize total rank. If we can't find 11, we use 9.
        # Schoolbook for 2x3x3 is 9.
        sol1 = None # Mark as needing schoolbook fallback
    
    # If sol1 is found, we use it. Otherwise, we use schoolbook.
    # Actually, if we can't find rank 11, we should try rank 9 (schoolbook) which is always possible.
    # But rank 9 is higher than 11. So if we find 11, we save 2 ranks per block.
    
    # Let's just use schoolbook for all if search fails, to ensure validity.
    # But we want to try to beat 48.
    
    # If we find decompositions for the blocks, we combine them.
    # U_total: (6, R_total)
    # V_total: (24, R_total)
    # W_total: (16, R_total)
    
    # We will construct the matrices block by block.
    U_list = []
    V_list = []
    W_list = []
    
    current_col_offset = 0
    for p_block in [3, 3, 2]:
        if time.time() > deadline:
            break
            
        # Determine rank for this block
        if p_block == 3:
            target_rank = 11
        else: # p_block == 2
            target_rank = 10
            
        sol = solve_block(n, m, p_block, target_rank)
        
        if sol is not None:
            U_b, V_b, W_b = sol
            R_b = U_b.shape[1]
        else:
            # Fallback to schoolbook for this block
            R_b = n * m * p_block
            U_b = np.zeros((n*m, R_b))
            V_b = np.zeros((m*p_block, R_b))
            W_b = np.zeros((p_block*n, R_b))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p_block):
                        U_b[i*m+j, col] = 1
                        V_b[j*p_block+k, col] = 1
                        W_b[k*n+i, col] = 1
                        col += 1
        
        # Append to lists
        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)
        
        # Update offsets for V and W indices?
        # V index is j * p_total + k.
        # In the block, V_b index is j * p_block + k.
        # We need to map V_b to V_total.
        # V_total has shape (m*p_total, R_total).
        # This is tricky because V_b is for a sub-range of k.
        # V_b[j * p_block + k] corresponds to V_total[j * p_total + (current_col_offset + k)].
        
        # We will construct the full matrices at the end.
        # Store (U_b, V_b, W_b, p_block, offset)
        blocks.append((U_b, V_b, W_b, p_block, current_col_offset))
        current_col_offset += p_block
        
    if time.time() > deadline:
        # Return schoolbook for everything
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Combine blocks
    # Total Rank is sum of R_b
    R_total = sum(U_b.shape[1] for U_b, _, _, _, _ in blocks)
    
    U_total = np.zeros((n*m, R_total))
    V_total = np.zeros((m*p, R_total))
    W_total = np.zeros((p*n, R_total))
    
    col_offset = 0
    for U_b, V_b, W_b, p_block, k_offset in blocks:
        R_b = U_b.shape[1]
        
        # U is same for all blocks (depends on i,j)
        U_total[:, col_offset:col_offset+R_b] = U_b
        
        # V_b has shape (m*p_block, R_b).
        # V_total has shape (m*p, R_total).
        # Row j * p_block + k in V_b corresponds to row j * p + (k_offset + k) in V_total.
        for j in range(m):
            for k in range(p_block):
                v_row_b = j * p_block + k
                v_row_t = j * p + (k_offset + k)
                V_total[v_row_t, col_offset:col_offset+R_b] = V_b[v_row_b]
                
        # W_b has shape (p_block*n, R_b).
        # W_total has shape (p*n, R_total).
        # Row k * n + i in W_b corresponds to row (k_offset + k) * n + i in W_total.
        for k in range(p_block):
            for i in range(n):
                w_row_b = k * n + i
                w_row_t = (k_offset + k) * n + i
                W_total[w_row_t, col_offset:col_offset+R_b] = W_b[w_row_b]
                
        col_offset += R_b
        
    return {"U": U_total, "V": V_total, "W": W_total}
