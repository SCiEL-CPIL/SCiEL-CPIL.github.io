import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Pre-compute indices for all possible equations
    num_eqs_total = n * n * p * p * m * m
    
    I = np.zeros(num_eqs_total, dtype=int)
    K = np.zeros(num_eqs_total, dtype=int)
    I2 = np.zeros(num_eqs_total, dtype=int)
    K2 = np.zeros(num_eqs_total, dtype=int)
    J = np.zeros(num_eqs_total, dtype=int)
    J2 = np.zeros(num_eqs_total, dtype=int)
    
    idx = 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for k2 in range(p):
                    for j in range(m):
                        for j2 in range(m):
                            I[idx] = i
                            K[idx] = k
                            I2[idx] = i2
                            K2[idx] = k2
                            J[idx] = j
                            J2[idx] = j2
                            idx += 1
                            
    b_full = np.zeros(num_eqs_total)
    mask = (I2 == I) & (K2 == K) & (J == J2)
    b_full[mask] = 1.0
    
    U_idx = I2 * m + J
    V_idx = J2 * p + K2
    W_col_base = K * n + I
    
    # Search for lower rank
    for R in range(rank_sb - 1, 69, -1):
        if time.time() > deadline - 2.0:
            break
            
        num_vars = p * n * R
        max_trials = 500  # Increased trials
        
        for trial in range(max_trials):
            if time.time() > deadline - 2.0:
                break
                
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            if num_vars <= num_eqs_total:
                eq_indices = np.random.choice(num_eqs_total, num_vars, replace=False)
            else:
                eq_indices = np.arange(num_eqs_total)
                
            b_sub = b_full[eq_indices]
            
            # Vectorized construction of A_sub
            # For each row i: eq = eq_indices[i], w_col = W_col_base[eq]
            # cols = r*(p*n) + w_col for r in 0..R-1
            # A_sub[i, cols] = U[U_idx[eq], r] * V[V_idx[eq], r]
            
            # Get all the needed indices
            eqs = eq_indices
            u_rows = U_idx[eqs]  # shape (num_vars,)
            v_rows = V_idx[eqs]  # shape (num_vars,)
            w_cols = W_col_base[eqs]  # shape (num_vars,)
            
            # For each row i and each r, the column is r*(p*n) + w_cols[i]
            # We need to scatter U[u_rows[i], r] * V[v_rows[i], r] into A_sub[i, r*(p*n)+w_cols[i]]
            
            # Build the column indices matrix: shape (num_vars, R)
            # col_indices[i, r] = r * (p*n) + w_cols[i]
            r_range = np.arange(R)
            col_indices = r_range[None, :] * (p*n) + w_cols[:, None]  # shape (num_vars, R)
            
            # Get the U and V values: shape (num_vars, R)
            u_vals = U[u_rows, :]  # shape (num_vars, R)
            v_vals = V[v_rows, :]  # shape (num_vars, R)
            prod_vals = u_vals * v_vals  # shape (num_vars, R)
            
            # Scatter into A_sub
            A_sub = np.zeros((num_vars, num_vars))
            row_indices = np.arange(num_vars)[:, None]  # shape (num_vars, 1)
            A_sub[row_indices, col_indices] = prod_vals
            
            try:
                w_sol = np.linalg.solve(A_sub, b_sub)
            except np.linalg.LinAlgError:
                continue
                
            w2 = w_sol * 2
            w2_rounded = np.round(w2)
            if not np.allclose(w2, w2_rounded, atol=1e-9):
                continue
                
            # Verification
            if num_vars < num_eqs_total:
                verify_indices = np.random.choice(num_eqs_total, num_vars, replace=False)
            else:
                verify_indices = np.arange(num_eqs_total)
                
            eqs_v = verify_indices
            u_rows_v = U_idx[eqs_v]
            v_rows_v = V_idx[eqs_v]
            w_cols_v = W_col_base[eqs_v]
            
            col_indices_v = r_range[None, :] * (p*n) + w_cols_v[:, None]
            u_vals_v = U[u_rows_v, :]
            v_vals_v = V[v_rows_v, :]
            prod_vals_v = u_vals_v * v_vals_v
            
            A_verify = np.zeros((len(verify_indices), num_vars))
            row_indices_v = np.arange(len(verify_indices))[:, None]
            A_verify[row_indices_v, col_indices_v] = prod_vals_v
            
            b_verify = b_full[verify_indices]
            residual = A_verify @ w_sol - b_verify
            if np.max(np.abs(residual)) > 1e-9:
                continue
                
            W_sol = w_sol.reshape((p*n, R))
            
            if R < best_rank:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W_sol
                break
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
