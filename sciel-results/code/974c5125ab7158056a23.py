# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the area of the union of 64 Kakeya needle triangles.
    Combines vectorized gradient-based optimization (L-BFGS-B) with
    Nelder-Mead polishing for non-smoothness and random restarts.
    """
    n = 64
    M = 20000
    
    # Precompute constants
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    # Initial guess: Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit
    
    def objective_and_grad(x):
        """
        Computes area (mean union length) and its gradient.
        x: shape (n,)
        """
        # L shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        S = np.sort(L, axis=1)
        
        D = np.diff(S, axis=1) # shape (M, n-1)
        G = np.minimum(D, w[:, None]) # shape (M, n-1)
        
        area = np.mean(w + np.sum(G, axis=1))
        
        # Gradient computation
        P = np.argsort(L, axis=1) # shape (M, n)
        
        # T[m, j] = 1 if D[m, j] < w[m]
        T = (D < w[:, None]).astype(np.float64) # shape (M, n-1)
        
        P_plus = P[:, 1:].ravel()
        P_minus = P[:, :-1].ravel()
        T_flat = T.ravel()
        
        grad_plus = np.bincount(P_plus, weights=T_flat, minlength=n)
        grad_minus = np.bincount(P_minus, weights=T_flat, minlength=n)
        
        grad_x = (grad_plus - grad_minus) / M
        
        return area, grad_x

    def obj_func(x):
        val, _ = objective_and_grad(x)
        return val
        
    def jac_func(x):
        _, grad = objective_and_grad(x)
        return grad

    best_x = x0.copy()
    best_area = np.inf
    
    # Evaluate initial point
    try:
        best_area, _ = objective_and_grad(best_x)
    except Exception:
        best_area = np.inf

    # 1. Main L-BFGS optimization
    if time.time() < deadline - 100:
        try:
            res = minimize(
                obj_func, 
                best_x, 
                method='L-BFGS-B', 
                jac=jac_func, 
                options={'maxiter': 2000, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() < deadline:
                if res.fun < best_area:
                    best_x = res.x
                    best_area = res.fun
        except Exception:
            pass

    # 2. Nelder-Mead Polish (handles non-smoothness)
    if time.time() < deadline - 50:
        try:
            res_nm = minimize(
                obj_func, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if time.time() < deadline:
                if res_nm.fun < best_area:
                    best_x = res_nm.x
                    best_area = res_nm.fun
        except Exception:
            pass

    # 3. Random Restarts
    np.random.seed(42)
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
            
        # Perturb current best
        noise_scale = 0.1 + 0.05 * restart
        x_start = best_x + noise_scale * np.random.randn(n)
        
        try:
            res = minimize(
                obj_func, 
                x_start, 
                method='L-BFGS-B', 
                jac=jac_func, 
                options={'maxiter': 300, 'maxfun': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() > deadline - 30:
                break
                
            if res.fun < best_area:
                best_x = res.x
                best_area = res.fun
                
                # Polish with Nelder-Mead if time permits
                if time.time() < deadline - 30:
                    try:
                        res_nm = minimize(
                            obj_func, 
                            best_x, 
                            method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
                        )
                        if time.time() < deadline:
                            if res_nm.fun < best_area:
                                best_x = res_nm.x
                                best_area = res_nm.fun
                    except Exception:
                        pass
        except Exception:
            pass

    # Final validation
    if not np.all(np.isfinite(best_x)) or best_area <= 0:
        return x0

    return np.array(best_x, dtype=np.float64)

# EVOLVE-BLOCK-END
