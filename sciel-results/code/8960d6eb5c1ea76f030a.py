import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq_from_coeffs(c):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        c: array of 13 Laguerre coefficients.
        """
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def coeffs_to_roots_penalty(c, roots):
        """
        Penalty for the constraint that the polynomial has double roots at the given locations.
        P(r_i) = 0 and P'(r_i) = 0 for all i.
        Returns sum of squared residuals.
        """
        penalty = 0.0
        for r in roots:
            # Evaluate P(r) = sum c_k L_k^{(1)}(r)
            p_val = 0.0
            p_deriv = 0.0
            for k in range(13):
                if c[k] == 0:
                    continue
                lk = eval_genlaguerre(k, 1, r)
                p_val += c[k] * lk
                # Derivative of L_k^{(1)}(x) = -L_{k-1}^{(2)}(x)
                if k >= 1:
                    lk_deriv = -eval_genlaguerre(k-1, 2, r)
                else:
                    lk_deriv = 0.0
                p_deriv += c[k] * lk_deriv
            
            penalty += p_val**2 + p_deriv**2
        
        return penalty

    def cost_func(params):
        """
        Minimize the upper bound on C4.
        params: [c_0, ..., c_12, r_0, ..., r_11]
        Total 25 parameters.
        
        The objective is 1/||P||^2 + lambda * penalty
        """
        c = params[:13]
        log_roots = params[13:]
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        if np.any(np.isnan(c)) or np.any(np.isinf(c)):
            return 1e10
            
        norm_sq = compute_norm_sq_from_coeffs(c)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        # Penalty for double root constraint
        penalty = coeffs_to_roots_penalty(c, roots)
        
        # Weight the penalty heavily to enforce the constraint
        # The norm should be O(1) to O(100), penalty should be near 0
        lambda_pen = 100.0
        
        return 1.0 / norm_sq + lambda_pen * penalty

    # Optimization Strategy:
    # 1. Global search over 25 parameters (13 coeffs + 12 log-roots) using Differential Evolution.
    # 2. Multiple restarts to avoid local minima.
    # 3. Final polish with L-BFGS-B.
    
    # Initial guess: 
    # For the seed roots r_i = 4 * 2^i, compute the corresponding c.
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    coeffs_desc = np.poly(seed_roots)
    p = coeffs_desc[::-1] # ascending
    seed_c = A_T_inv @ p
    
    seed_log_roots = np.log(seed_roots)
    seed_params = np.concatenate([seed_c, seed_log_roots])
    
    # Bounds for parameters
    # Coefficients: allow a wide range
    # Roots: log range [-4, 10]
    bounds = [(-100.0, 100.0)] * 13 + [(-4.0, 10.0)] * 12
    
    best_params = None
    best_val = 1e10
    best_c = None
    best_roots = None

    # Run Differential Evolution with multiple restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for polish and fallback
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # 25D is higher, so be conservative
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                best_c = result.x[:13]
                best_roots = np.exp(result.x[13:])
                
        except Exception as e:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    seed_val = cost_func(seed_params)
    
    if best_params is None or seed_val < best_val:
        best_val = seed_val
        best_params = seed_params
        best_c = seed_c
        best_roots = seed_roots

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_params is not None:
        try:
            polish_result = minimize(
                cost_func, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_params = polish_result.x
                best_c = polish_result.x[:13]
                best_roots = np.exp(polish_result.x[13:])
        except Exception:
            pass

    # Extract roots from the best solution
    # If we have best_roots, use them. Otherwise, use seed.
    if best_roots is not None:
        # Verify the roots produce a valid polynomial with the given coefficients
        # If the penalty is high, the roots might not be accurate.
        # For the final output, we need to return 12 positive double-root locations.
        # The task says "construct() -> list of 12 positive double-root locations".
        # So we return the roots.
        
        # Ensure roots are positive
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
