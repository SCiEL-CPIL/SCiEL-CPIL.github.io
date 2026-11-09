import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1

    # Precompute A matrix
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
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

    def params_to_roots(params):
        """
        params: u_1, ..., u_12
        r_1 = exp(u_1)
        r_i = r_{i-1} * exp(u_i) for i > 1
        """
        roots = np.zeros(12)
        roots[0] = np.exp(params[0])
        for i in range(1, 12):
            roots[i] = roots[i-1] * np.exp(params[i])
        return roots

    def cost_func(params):
        roots = params_to_roots(params)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Bounds for params
    # u_1: log(r_1). r_1 in [0.01, 10000] -> [-4.6, 9.2]
    # u_i (i>1): log(r_i/r_{i-1}). Ratio in [1, 100] -> [0, 4.6]
    bounds = [(-4.6, 9.2)] + [(0.0, 4.6)] * 11

    best_val = 1e10
    best_params = None

    # Strategy: Dual Annealing is often better for global optimization with bounds
    # and handles the smoothness well.
    
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        try:
            result = dual_annealing(
                cost_func, 
                bounds, 
                seed=seed,
                maxiter=100, # Dual annealing is slower per iter?
                restarts=5,
                polish=True
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception as e:
            pass
            
    # Fallback to DE if DA fails or is slow
    if best_params is None or time.time() - start_time < deadline - 100:
        # Run DE as well or instead?
        # Let's stick to one strong optimizer to save time.
        # If DA didn't find anything, try DE.
        if best_val == 1e10:
             try:
                result = differential_evolution(
                    cost_func, 
                    bounds, 
                    seed=42, 
                    maxiter=200, 
                    popsize=20, 
                    tol=1e-10, 
                    polish=True, 
                    init='sobol'
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
             except Exception:
                pass

    # Final Polish
    if best_params is not None and time.time() - start_time < deadline - 10:
        try:
            polish_result = minimize(
                cost_func, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_params = polish_result.x
        except Exception:
            pass

    if best_params is not None:
        best_roots = params_to_roots(best_params)
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
