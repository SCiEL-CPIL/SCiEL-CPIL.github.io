import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 100.0  # Leave margin for 120s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def error_func(params, r):
        # params is a flat array of size r * (nm + mp + pn)
        nm, mp, pn = n * m, m * p, p * n
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp :]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        # Compute sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want this to be 1 for all i, j, k.
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    err += (s - 1.0) ** 2
        return err

    def check_solution(U, V, W, r):
        # Check if entries are close to half-integers
        # And if the decomposition is exact
        for c in range(r):
            for val in U[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in V[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in W[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
        
        # Exact check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find lower rank
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # We try ranks from 23 down to 15
    for r in range(rank_sb - 1, 14, -1):
        if time.time() - start_time > max_time:
            break
            
        nm, mp, pn = n * m, m * p, p * n
        total_vars = r * (nm + mp + pn)
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > max_time:
                break
                
            # Random initialization
            # Scale by 1 to keep values small
            x0 = np.random.uniform(-1, 1, size=total_vars)
            
            # Use Nelder-Mead
            # This might be slow, so we limit iterations
            from scipy.optimize import minimize
            try:
                res = minimize(error_func, x0, args=(r,), method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                continue
                
            if res.success and res.fun < 1e-8:
                # Extract U, V, W
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp :]
                
                U = U_flat.reshape(r, nm)
                V = V_flat.reshape(r, mp)
                W = W_flat.reshape(r, pn)
                
                if check_solution(U, V, W, r):
                    # Convert to lists of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Round to half-integers
                    U_list = [[round(val * 2) / 2.0 for val in row] for row in U_list]
                    V_list = [[round(val * 2) / 2.0 for val in row] for row in V_list]
                    W_list = [[round(val * 2) / 2.0 for val in row] for row in W_list]
                    
                    # Final verification
                    if check_solution(np.array(U_list), np.array(V_list), np.array(W_list), r):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
    return best_result
