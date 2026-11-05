import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=tol, rtol=tol)

    def error_func(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        return np.dot(diff.ravel(), diff.ravel())

    def error_grad(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        
        grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        
        return np.dot(diff.ravel(), diff.ravel()), grad

    def polish(params, R, max_iter=200):
        """High-precision polish pass after initial convergence."""
        try:
            res = minimize(error_func, params, args=(R,), method='L-BFGS-B', 
                           jac=error_grad,
                           options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': max_iter * 2})
            return res.x
        except Exception:
            return params

    def solve_rank(R, max_restarts=50):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        best_params = None
        best_fun = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                # Start from truncated schoolbook + perturbation
                U0 = U_fb[:, :R].copy()
                V0 = V_fb[:, :R].copy()
                W0 = W_fb[:, :R].copy()
                U0 += np.random.randn(size_u, R) * 0.1
                V0 += np.random.randn(size_v, R) * 0.1
                W0 += np.random.randn(size_w, R) * 0.1
                params0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
            elif restart % 3 == 1:
                params0 = np.random.randn(total_size) * 0.5
            else:
                params0 = np.random.randn(total_size) * 1.0
            
            try:
                res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                               jac=error_grad,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000})
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_params = res.x.copy()
                
                if res.fun < 1e-5:
                    # Polish for high precision
                    polished = polish(res.x, R)
                    U = polished[:size_u*R].reshape(size_u, R)
                    V = polished[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = polished[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    # Try snapping to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                    
                    # Try snapping to integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify(U_int, V_int, W_int):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    
                    # Also try without snapping
                    if verify(U, V, W, tol=1e-7):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
            except Exception:
                pass
        
        # Final attempt with best params if we got close
        if best_params is not None and best_fun < 1e-3:
            try:
                polished = polish(best_params, R, max_iter=500)
                U = polished[:size_u*R].reshape(size_u, R)
                V = polished[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                W = polished[size_u*R + size_v*R :].reshape(size_w, R)
                
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify(U_snap, V_snap, W_snap):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
            except Exception:
                pass
                
        return None

    # Try to find lower ranks, starting from lower ranks
    for R in range(30, 45):
        if time.time() > deadline - 10:
            break
        
        # Distribute time proportionally
        remaining_ranks = 44 - R
        time_budget = max(5, int((deadline - time.time()) / max(1, remaining_ranks)))
        est_restarts = max(5, min(40, int(time_budget / 0.3)))
        
        result = solve_rank(R, max_restarts=est_restarts)
        if result is not None:
            return result

    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
