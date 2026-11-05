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
                    val = np.dot(U[idx_u], V[idx_v]) # This is not right.
                    # The term is sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    # This is element-wise product then sum.
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

    # Search for lower rank
    # We try ranks from rank_school - 1 down to some limit
    # Time limit 120s.
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Lower bound is weak, but let's try a few ranks below 48
    # If we find one, we return it.
    
    # To speed up search, we can use random restarts and local minimization.
    # Objective: minimize sum_{i,j,k} (1 - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])^2
    
    def objective(params, R):
        # params is a flat vector of length R*(nm + mp + pn)
        nm = n * m
        mp = m * p
        pn = p * n
        size_u = nm * R
        size_v = mp * R
        size_w = pn * R
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:size_u+size_v+size_w]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
                    # Vectorized dot product of element-wise products
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    err += (1.0 - val) ** 2
        return err

    # We will try R = 47, 46, 45 ...
    # For each R, try a few random starts.
    
    # Optimization: Since the problem is small, we can use scipy.optimize if available, 
    # but simple gradient descent or Nelder-Mead might be safer/faster to implement without imports if scipy is flaky.
    # However, scipy is available.
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None

    best_result = None
    
    for R in range(rank_school - 1, max(rank_school - 10, 30), -1):
        if time.time() - start_time > time_limit:
            break
            
        nm = n * m
        mp = m * p
        pn = p * n
        total_vars = R * (nm + mp + pn)
        
        # Try a few random initializations
        attempts = 5
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Random init
            params = np.random.normal(0.1, 0.5, total_vars)
            
            if minimize:
                res = minimize(objective, params, args=(R,), method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success or res.fun < 1e-4:
                    # Check if we can round to half-integers and get exact solution
                    p_opt = res.x
                    U_opt = p_opt[:nm*R].reshape(nm, R)
                    V_opt = p_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = p_opt[nm*R+mp*R:].reshape(pn, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if is_valid_exact(U_round, V_round, W_round):
                        best_result = (R, U_round, V_round, W_round)
                        break
            else:
                # Fallback: simple local search or just skip
                pass
                
        if best_result:
            break

    if best_result:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
