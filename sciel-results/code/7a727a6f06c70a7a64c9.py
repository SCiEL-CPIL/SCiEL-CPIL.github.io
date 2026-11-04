import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    # Build target tensor
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc, Target, atol=tol)

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)

    def solve_for_W(U, V):
        """Given U and V, solve for W using least squares."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        R = U_arr.shape[1]
        
        W = np.zeros((dim_W, R))
        residual = 0.0
        
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        
        for gamma in range(dim_W):
            A = M.reshape(dim_U * dim_V, R)
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            r = b - A @ x
            residual += np.sum(r**2)
        
        return W, residual

    def objective(params):
        """Objective: minimize residual of W least-squares."""
        U_size = dim_U * R
        U_flat = params[:U_size]
        V_flat = params[U_size:]
        
        U = U_flat.reshape(dim_U, R)
        V = V_flat.reshape(dim_V, R)
        
        _, res = solve_for_W(U, V)
        return res

    start_time = time.time()
    time_limit = 110.0

    # Schoolbook solution (rank 75)
    best_result = None
    best_rank = n * m * p  # 75
    
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks using optimization
    # Try ranks from 25 (lower bound) up to 74
    for R in range(25, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # More restarts for lower ranks
        n_restarts = max(30, 200 - 2*R)
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small half-integers
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, R))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, R))
            
            # Check initial residual
            W_check, res_check = solve_for_W(U_init, V_init)
            if res_check < 1e-12:
                if is_half_integer(W_check, tol=1e-6):
                    W_snap = snap_to_half_int(W_check)
                    if verify(U_init, V_init, W_snap):
                        best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_snap.tolist()}
                        best_rank = R
                        found = True
                        break
            
            # Optimize U and V
            params_init = np.concatenate([U_init.flatten(), V_init.flatten()])
            
            try:
                res = minimize(objective, params_init, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-12:
                params_opt = res.x
                U_opt = params_opt[:dim_U*R].reshape(dim_U, R)
                V_opt = params_opt[dim_U*R:].reshape(dim_V, R)
                
                W_opt, res_final = solve_for_W(U_opt, V_opt)
                
                if res_final < 1e-12:
                    # Try to snap to half-integers
                    if is_half_integer(W_opt, tol=1e-6):
                        W_snap = snap_to_half_int(W_opt)
                        if verify(U_opt, V_opt, W_snap):
                            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_snap.tolist()}
                            best_rank = R
                            found = True
                            break
                    # Also try snapping U and V
                    U_snap = snap_to_half_int(U_opt)
                    V_snap = snap_to_half_int(V_opt)
                    W2, res2 = solve_for_W(U_snap, V_snap)
                    if res2 < 1e-12 and is_half_integer(W2, tol=1e-6):
                        W2_snap = snap_to_half_int(W2)
                        if verify(U_snap, V_snap, W2_snap):
                            best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W2_snap.tolist()}
                            best_rank = R
                            found = True
                            break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
