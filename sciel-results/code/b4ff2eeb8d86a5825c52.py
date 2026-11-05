import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Precompute the target tensor values
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We represent the decomposition condition:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Helper to check if a set of factors is a valid exact decomposition
    def is_valid_exact(U, V, W):
        # U, V, W are numpy arrays of shape (nm, R), (mp, R), (pn, R)
        # Check integer/half-integer
        # Multiply by 2 to check integer-ness
        U2 = 2 * U
        V2 = 2 * V
        W2 = 2 * W
        
        # Check if close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False
            
        # Check exact decomposition
        # We need to compute sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] for all i,j,k
        # This is a 3-way contraction.
        # We can do it by iterating i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    prod = U[idx_u] * V[idx_v] * W[idx_w]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Helper to construct schoolbook decomposition
    def schoolbook():
        R = rank_school
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try structured decompositions based on algebraic insights
    # For 3x4 * 4x4, we can try block-based decompositions
    
    def try_block_decomposition():
        """Try decompositions that exploit the rectangular structure"""
        # Strategy: group indices to reduce rank
        # For n=3, m=4, p=4:
        # We can try to share computations across blocks
        
        # Approach 1: Use the fact that 3 and 4 have some structure
        # Try rank = 3*4 + 4*4 - overlap = 12 + 16 - overlap
        
        # Let's try a systematic search over structured patterns
        # Pattern: U depends on (i,j), V on (j,k), W on (k,i)
        
        # Try various rank values and see if we can construct valid decompositions
        for R in range(32, 48):  # Try ranks from 32 to 47
            # Try to construct a decomposition with this rank
            # Using a greedy approach with structured initialization
            
            # Initialize with zeros
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is a system of 48 equations in R*12 + R*16 + R*12 = R*40 variables
            
            # Use a simple iterative method: for each (i,j,k), try to assign values
            # to satisfy the equation using unused or underutilized rank components
            
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        
                        # Current contribution
                        current = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                        needed = 1.0 - current
                        
                        if abs(needed) < 1e-10:
                            continue
                        
                        # Find a rank component r where we can add a small correction
                        # Try each rank component
                        found = False
                        for r in range(R):
                            # Check if this component is "available" (not heavily used elsewhere)
                            # Simple heuristic: check if adding to this component helps
                            u_val = U[idx_u, r]
                            v_val = V[idx_v, r]
                            w_val = W[idx_w, r]
                            
                            # If all three are zero, we can set one of them
                            if u_val == 0 and v_val == 0 and w_val == 0:
                                # Set U to 1, V to 1, W to needed
                                U[idx_u, r] = 1.0
                                V[idx_v, r] = 1.0
                                W[idx_w, r] = needed
                                found = True
                                break
                            elif u_val == 0 and v_val != 0 and w_val != 0:
                                # Set U to needed / (v_val * w_val)
                                if abs(v_val * w_val) > 1e-10:
                                    U[idx_u, r] = needed / (v_val * w_val)
                                    found = True
                                    break
                            elif u_val != 0 and v_val == 0 and w_val != 0:
                                if abs(u_val * w_val) > 1e-10:
                                    V[idx_v, r] = needed / (u_val * w_val)
                                    found = True
                                    break
                            elif u_val != 0 and v_val != 0 and w_val == 0:
                                if abs(u_val * v_val) > 1e-10:
                                    W[idx_w, r] = needed / (u_val * v_val)
                                    found = True
                                    break
                            elif u_val != 0 and v_val != 0 and w_val != 0:
                                # Adjust one of them
                                # Try adjusting W
                                if abs(u_val * v_val) > 1e-10:
                                    W[idx_w, r] += needed / (u_val * v_val)
                                    found = True
                                    break
                        
                        if not found:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                # Check if it's valid
                if is_valid_exact(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Search for lower rank
    # We try ranks from rank_school - 1 down to some limit
    # Time limit 120s.
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # First try structured approaches
    result = try_block_decomposition()
    if result:
        return result
    
    # If structured approach fails, try random search with optimization
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    best_result = None
    
    # Try ranks from 40 down to 35
    for R in range(40, 34, -1):
        if time.time() - start_time > time_limit:
            break
            
        nm = n * m
        mp = m * p
        pn = p * n
        total_vars = R * (nm + mp + pn)
        
        # Try a few random initializations
        attempts = 3
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Random init with small values
            params = np.random.normal(0.0, 0.3, total_vars)
            
            if minimize:
                # Define objective
                def objective(p_vec, R_val):
                    U_flat = p_vec[:nm * R_val]
                    V_flat = p_vec[nm * R_val:nm * R_val + mp * R_val]
                    W_flat = p_vec[nm * R_val + mp * R_val:]
                    
                    U_mat = U_flat.reshape(nm, R_val)
                    V_mat = V_flat.reshape(mp, R_val)
                    W_mat = W_flat.reshape(pn, R_val)
                    
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.sum(U_mat[idx_u] * V_mat[idx_v] * W_mat[idx_w])
                                err += (1.0 - val) ** 2
                    return err
                
                res = minimize(objective, params, args=(R,), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.fun < 1e-3:
                    # Check if we can round to half-integers
                    p_opt = res.x
                    U_opt = p_opt[:nm * R].reshape(nm, R)
                    V_opt = p_opt[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_opt = p_opt[nm * R + mp * R:].reshape(pn, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if is_valid_exact(U_round, V_round, W_round):
                        best_result = (R, U_round, V_round, W_round)
                        break
            else:
                # Fallback: skip
                pass
                
        if best_result:
            break

    if best_result:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
