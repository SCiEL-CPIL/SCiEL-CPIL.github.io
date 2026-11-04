"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    # c = A_T_inv @ p where p is standard-basis coeffs (ascending)
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute squared Laguerre norm of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # Convert to Laguerre basis
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize upper bound on C4 ~ 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_with_grad(log_roots):
        """
        Cost and analytic gradient using polynomial division.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros_like(log_roots)
        
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # Convert to Laguerre basis
        c = A_T_inv @ p
        
        # Norm squared
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros_like(log_roots)
        
        cost = 1.0 / norm_sq
        
        # Compute gradient analytically
        grad = np.zeros(12)
        
        for i in range(12):
            other_roots = np.delete(roots, i)
            coeffs_other_desc = np.poly(other_roots)
            p_other = coeffs_other_desc[::-1]  # ascending
            
            dp_dr_i = -p_other
            dc_dr_i = A_T_inv @ dp_dr_i
            dnorm_sq_dr_i = 2.0 * np.sum(c * (np.arange(13) + 1) * dc_dr_i)
            dcost_dr_i = -dnorm_sq_dr_i / (norm_sq * norm_sq)
            grad[i] = dcost_dr_i * roots[i]
        
        return cost, grad

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Final high-precision polish with L-BFGS-B using analytic gradient.
    
    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-2.0, 8.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Run Differential Evolution with diverse restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=20, 
                tol=1e-10, 
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback / Initial guess
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()
        best_log_roots = seed_log_roots.copy()

    # Final high-precision polish with L-BFGS-B using analytic gradient
    if best_log_roots is not None and time_ok():
        try:
            polish_result = minimize(
                cost_with_grad,
                best_log_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 50}
            )
            
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
    
    # Additional polish pass with even tighter tolerances if time permits
    if best_log_roots is not None and time_ok() and (time.time() - start_time) < (DEADLINE - 60.0):
        try:
            polish_result2 = minimize(
                cost_with_grad,
                best_log_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100}
            )
            
            if polish_result2.fun < best_val:
                best_val = polish_result2.fun
                best_log_roots = polish_result2.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
