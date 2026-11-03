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
    
    # Target tensor for M_{2,3,2} is all ones
    # T[i,j,k] = 1
    
    def error_func_vec(params, r):
        # params shape: 160 for r=10
        U_flat = params[:sub_n*sub_m*r]
        V_flat = params[sub_n*sub_m*r : sub_n*sub_m*r + sub_m*sub_p*r]
        W_flat = params[sub_n*sub_m*r + sub_m*sub_p*r :].reshape(sub_p*sub_n, r)
        
        U = U_flat.reshape(sub_n, sub_m, r)
        V = V_flat.reshape(sub_m, sub_p, r)
        W = W_flat.reshape(sub_p, sub_n, r)
        
        # We want T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Dimensions:
        # U: (2, 3, r)
        # V: (3, 2, r)
        # W: (2, 2, r)
        
        # Broadcast to (2, 3, 2, r)
        # U needs broadcast on k (dim 2)
        # V needs broadcast on i (dim 0)
        # W needs broadcast on j (dim 1)
        
        # Use einsum
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # np.einsum('ijc,jkc,kic->ijk', U, V, W)
        
        T_calc = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        
        # Target is all ones
        err = np.sum((T_calc - 1.0)**2)
        return err

    def verify_sub(U, V, W):
        r = U.shape[1]
        # Reshape to 3D
        U3 = U.reshape(sub_n, sub_m, r)
        V3 = V.reshape(sub_m, sub_p, r)
        W3 = W.reshape(sub_p, sub_n, r)
        
        T_calc = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if not np.allclose(T_calc, 1.0, atol=1e-9):
            return False
        return True

    def find_sub_decomposition(r, max_restarts=50, timeout_per_restart=0.5):
        best_err = float('inf')
        best_params = None
        
        total_vars = sub_n*sub_m*r + sub_m*sub_p*r + sub_p*sub_n*r
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            # Scale matters. Values around 1 are good.
            params0 = np.random.randn(total_vars) * 0.5
            
            try:
                res = opt.minimize(error_func_vec, params0, args=(r,), method='L-BFGS-B', 
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

    # Try ranks 10, 11
    # Rank 10 is the target.
    for r_sub in [10, 11]:
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
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                
                T_calc = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                
                if not np.allclose(T_calc, 1.0, atol=1e-9):
                    return False
                return True
            
            if verify_full(U_full, V_full, W_full):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                
    return best_result
