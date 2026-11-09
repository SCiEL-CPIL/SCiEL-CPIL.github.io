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
    # Equation index: (i, k, i2, k2, j, j2)
    # Total equations: n*n*p*p*m*m
    num_eqs_total = n * n * p * p * m * m
    
    # We store the indices needed to build A and b for any equation
    # Eq e corresponds to:
    # Sum_r W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r] = 1 if (i2==i, k2==k, j==j2) else 0
    
    # To speed up, we pre-compute the "row" part of the linear system for a given R.
    # But R changes. So we pre-compute the index mappings.
    
    # List of tuples (i, k, i2, k2, j, j2)
    # We can flatten these into arrays for vectorized access.
    
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
    
    # Pre-compute U_indices and V_indices for each equation
    # U index for eq e is I2[e]*m + J[e]
    # V index for eq e is J2[e]*p + K2[e]
    U_idx = I2 * m + J
    V_idx = J2 * p + K2
    
    # W column index for eq e is K[e]*n + I[e]
    W_col_base = K * n + I
    
    # Search for lower rank
    for R in range(rank_sb - 1, 69, -1):
        if time.time() > deadline - 2.0:
            break
            
        num_vars = p * n * R
        # We need to solve A_sub * w = b_sub where A_sub is (num_vars x num_vars)
        # We pick num_vars random equations.
        
        max_trials = 200 # Increased trials due to faster subset solve
        
        for trial in range(max_trials):
            if time.time() > deadline - 2.0:
                break
                
            # Random U and V
            # Using {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Pick random subset of equations
            # We need exactly num_vars equations.
            # If num_vars > num_eqs_total, we use all (and use lstsq)
            if num_vars <= num_eqs_total:
                eq_indices = np.random.choice(num_eqs_total, num_vars, replace=False)
            else:
                eq_indices = np.arange(num_eqs_total)
                # If we have more variables than equations, we can't solve uniquely.
                # But for R < 84, num_vars = 21R.
                # Max num_vars for R=83 is 1743.
                # num_eqs_total = 9*49*16 = 7056.
                # So we always have enough equations.
                
            # Extract sub-vectors for b
            b_sub = b_full[eq_indices]
            
            # Construct A_sub
            # A_sub[row, col] where col corresponds to W[k*n+i, r]
            # For equation e in eq_indices:
            # Coeff for W_col_base[e] + r*R is U[U_idx[e], r] * V[V_idx[e], r]
            
            # We can build A_sub vectorized.
            # A_sub has shape (num_vars, num_vars)
            # Rows correspond to eq_indices
            # Cols correspond to (r, w_col) pairs.
            
            # Let's flatten the column index: col = r * (p*n) + w_col
            # For a given row e (which is eq_indices[i]), the coefficients are:
            # coeff[r, w_col] = U[U_idx[e], r] * V[V_idx[e], r] if w_col == W_col_base[e] else 0
            
            # This means each row of A_sub is very sparse! Only R non-zeros per row.
            # We can exploit this.
            
            A_sub = np.zeros((num_vars, num_vars))
            
            # Vectorized construction of the sparse rows
            # For each row i in 0..num_vars-1:
            # eq = eq_indices[i]
            # w_col = W_col_base[eq]
            # u_row = U[U_idx[eq], :]
            # v_row = V[V_idx[eq], :]
            # prod = u_row * v_row  (shape R)
            # A_sub[i, r*(p*n) + w_col] = prod[r]
            
            # Let's do this in a loop for clarity and speed (num_vars is ~1500, so 1500 iterations is fast)
            for i in range(num_vars):
                eq = eq_indices[i]
                w_col = W_col_base[eq]
                u_row = U[U_idx[eq], :]
                v_row = V[V_idx[eq], :]
                prod = u_row * v_row
                # Scatter prod into A_sub[i, :]
                cols = np.arange(R) * (p*n) + w_col
                A_sub[i, cols] = prod
            
            try:
                w_sol = np.linalg.solve(A_sub, b_sub)
            except np.linalg.LinAlgError:
                continue
                
            # Check for half-integers
            w2 = w_sol * 2
            w2_rounded = np.round(w2)
            if not np.allclose(w2, w2_rounded, atol=1e-9):
                continue
                
            # Verification step: Check against a different set of equations
            # Pick another random subset of size min(num_vars, num_eqs_total)
            if num_vars < num_eqs_total:
                verify_indices = np.random.choice(num_eqs_total, num_vars, replace=False)
            else:
                verify_indices = np.arange(num_eqs_total)
                
            # Build A_verify and b_verify
            A_verify = np.zeros((len(verify_indices), num_vars))
            b_verify = b_full[verify_indices]
            
            for i in range(len(verify_indices)):
                eq = verify_indices[i]
                w_col = W_col_base[eq]
                u_row = U[U_idx[eq], :]
                v_row = V[V_idx[eq], :]
                prod = u_row * v_row
                cols = np.arange(R) * (p*n) + w_col
                A_verify[i, cols] = prod
                
            # Check residual
            residual = A_verify @ w_sol - b_verify
            if np.max(np.abs(residual)) > 1e-9:
                continue
                
            # Valid solution found
            W_sol = w_sol.reshape((p*n, R))
            
            if R < best_rank:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W_sol
                break # Found one for this R, try lower R
        
        # If we found a solution for R, we continue to try R-1
        # If we didn't, we continue to try R-1 (maybe it was just bad luck)
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
