import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 570

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Seed: geometric sequence
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)
    
    best_val = seed_val
    best_log_roots = seed_log_roots.copy()
    best_roots = seed_roots.copy()

    log_low = -2.0
    log_high = 12.0
    bounds = [(log_low, log_high)] * 12

    # Strategy 1: Differential Evolution
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=30, 
                tol=1e-14, 
                polish=False,
                init='sobol',
                disp=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_log_roots = de_result.x.copy()
                best_roots = np.exp(best_log_roots)
    except Exception:
        pass

    # Strategy 2: Local polish from best
    if time.time() < deadline - 10 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(res.x)
        except Exception:
            pass

    # Strategy 3: Nelder-Mead polish (no gradient needed, robust)
    if time.time() < deadline - 10 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func, 
                best_log_roots, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(res.x)
        except Exception:
            pass

    # Strategy 4: Multi-start with diverse initializations
    if time.time() < deadline - 20:
        rng = np.random.default_rng(99)
        for attempt in range(8):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Perturb best
                init = best_log_roots + rng.normal(0, 1.0, size=12)
            elif attempt % 3 == 1:
                # Random in bounds
                init = rng.uniform(log_low, log_high, size=12)
            else:
                # Geometric with different ratio
                ratio = rng.uniform(1.5, 3.0)
                base = rng.uniform(1.0, 10.0)
                init = np.log(base * (ratio ** np.arange(12)))
                init = np.clip(init, log_low, log_high)
            
            try:
                res = minimize(
                    cost_func, 
                    init, 
                    method='L-BFGS-B',
                    bounds=bounds, 
                    options={'maxiter': 2000, 'ftol': 1e-14}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(res.x)
            except Exception:
                pass

    # Final: one more polish cycle
    if time.time() < deadline - 5 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(res.x)
        except Exception:
            pass

    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    return [4.0 * (2.0 ** i) for i in range(12)]
