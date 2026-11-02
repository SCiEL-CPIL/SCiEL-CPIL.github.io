import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 3, 6
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    
    sub_n, sub_m, sub_p = 2, 3, 2
    
    # Target tensor for M_{2,3,2}
    T_sub = np.zeros((sub_n, sub_m, sub_p))
    for i in range(sub_n):
        for j in range(sub_m):
            for k in range(sub_p):
                if i == k:
                    T_sub[i, j, k] = 1.0
    
    def error_func(params, r):
        U = params[:sub_n*sub_m*r].reshape(sub_n*sub_m, r)
        V = params[sub_n*sub_m*r : sub_n*sub_m*r + sub_m*sub_p*r].reshape(sub_m*sub_p, r)
        W = params[sub_n*sub_m*r + sub_m*sub_p*r :].reshape(sub_p*sub_n, r)
        
        # Compute tensor
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Vectorized:
        # U_flat[i*m+j, c]
        # V_flat[j*p+k, c]
        # W_flat[k*n+i, c]
        
        errs = 0.0
        for i in range(sub_n):
            for j in range(sub_m):
                for k in range(sub_p):
                    u_idx = i * sub_m + j
                    v_idx = j * sub_p + k
                    w_idx = k * sub_n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :]) # This is wrong, W is also involved
                    # val = sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
                    # This is a tensor contraction.
                    # Let's do it simply for small dimensions.
                    prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                    val = np.sum(prod)
                    errs += (val - T_sub[i, j, k])**2
        return errs

    def find_sub_decomposition(r, max_restarts=20, timeout=2.0):
        best_err = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            params0 = np.random.randn(sub_n*sub_m*r + sub_m*sub_p*r + sub_p*sub_n*r) * 0.5
            
            # Optimize
            try:
                res = opt.minimize(error_func, params0, args=(r,), method='L-BFGS-B', 
                                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
                    
                    if best_err < 1e-9:
                        # Round and verify
                        U = best_params[:sub_n*sub_m*r].reshape(sub_n*sub_m, r)
                        V = best_params[sub_n*sub_m*r : sub_n*sub_m*r + sub_m*sub_p*r].reshape(sub_m*sub_p, r)
                        W = best_params[sub_n*sub_m*r + sub_m*sub_p*r :].reshape(sub_p*sub_n, r)
                        
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if verify_sub(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded, 0.0
            except Exception:
                pass
                
        if best_params is not None:
            U = best_params[:sub_n*sub_m*r].reshape(sub_n*sub_m, r)
            V = best_params[sub_n*sub_m*r : sub_n*sub_m*r + sub_m*sub_p*r].reshape(sub_m*sub_p, r)
            W = best_params[sub_n*sub_m*r + sub_m*sub_p*r :].reshape(sub_p*sub_n, r)
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify_sub(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded, 0.0
                
        return None, None, None, best_err

    def verify_sub(U, V, W):
        r = U.shape[1]
        for i in range(sub_n):
            for j in range(sub_m):
                for k in range(sub_p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*sub_m+j, c] * V[j*sub_p+k, c] * W[k*sub_n+i, c]
                    target = 1.0 if i == k else 0.0
                    if abs(val - target) > 1e-9:
                        return False
        return True

    # Try ranks 8, 9, 10, 11
    for r_sub in [8, 9, 10, 11]:
        if time.time() > deadline - 20:
            break
            
        U_s, V_s, W_s, err = find_sub_decomposition(r_sub)
        
        if U_s is not None:
            total_rank = r_sub * 3
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            for b in range(3):
                col_offset = b * r_sub
                
                # U
                U_full[:, col_offset:col_offset+r_sub] = U_s
                
                # V
                for j in range(sub_m):
                    for k_local in range(sub_p):
                        k_global = 2*b + k_local
                        V_full[j * p + k_global, col_offset:col_offset+r_sub] = V_s[j * sub_p + k_local, :]
                
                # W
                for k_local in range(sub_p):
                    for i in range(sub_n):
                        k_global = 2*b + k_local
                        W_full[k_global * n + i, col_offset:col_offset+r_sub] = W_s[k_local * sub_n + i, :]
            
            # Verify full
            def verify_full(U, V, W):
                r = U.shape[1]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1.0) > 1e-9:
                                return False
                return True
            
            if verify_full(U_full, V_full, W_full):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                
    return best_result
