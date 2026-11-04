"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
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
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

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

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_func_geometric(log_a, log_q):
        """
        Cost function for geometric sequence parameterization.
        roots_i = exp(log_a + i * log_q)
        """
        a = math.exp(log_a)
        q = math.exp(log_q)
        
        if a <= 0 or q <= 0:
            return 1e10
            
        roots = a * (q ** np.arange(12))
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 2D geometric parameters (log_a, log_q) using DE.
    # 2. Use the best geometric sequence as a starting point for a 12D L-BFGS-B polish.
    
    best_roots = None
    best_val = 1e10
    
    # Step 1: Search for the best geometric sequence
    # Bounds for log_a and log_q
    # Roots in the seed were 4 * 2^i. log_a ~ log(4) ~ 1.4, log_q ~ log(2) ~ 0.69
    # Let's allow a wide range.
    bounds_geo = [(-5.0, 5.0), (-1.0, 2.0)]
    
    # Run DE on the 2D geometric space
    # This is very fast, so we can afford many iterations/restarts
    seeds_geo = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds_geo:
        if not time_ok():
            break
        
        try:
            result_geo = differential_evolution(
                cost_func_geometric, 
                bounds_geo, 
                seed=seed, 
                maxiter=200, 
                popsize=20, 
                tol=1e-12, 
                polish=True,
                init='sobol'
            )
            
            if result_geo.fun < best_val:
                best_val = result_geo.fun
                log_a, log_q = result_geo.x
                # Convert to log_roots for the next step
                best_log_roots = log_a + np.arange(12) * log_q
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback initial guess if DE fails
    if best_roots is None:
        seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
        best_roots = np.exp(seed_log_roots)
        best_val = cost_func(seed_log_roots)
    else:
        # Ensure we have the log_roots array for the polish step
        # Reconstruct from best_roots if not saved properly, but we saved best_log_roots
        # If best_roots was set in the loop, best_log_roots is also set.
        # If fallback was used, best_log_roots is set there.
        pass

    # Step 2: Polish with 12D L-BFGS-B starting from the best geometric sequence
    # This allows the roots to deviate slightly from the perfect geometric progression
    # to find a potentially better local minimum.
    
    if time_ok():
        try:
            # Use the best geometric log_roots as the initial guess
            initial_log_roots = best_log_roots.copy()
            
            # Run L-BFGS-B
            result_polish = minimize(
                cost_func, 
                initial_log_roots, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            
            if result_polish.fun < best_val:
                best_val = result_polish.fun
                best_roots = np.exp(result_polish.x)
                
        except Exception:
            pass

    # Step 3: Multi-start 12D DE if time permits, using the geometric seed as a strong candidate
    # But 12D DE is slow. We might just run a few quick DEs with a small population
    # centered around the current best solution if it seems promising, 
    # or just rely on the polish. 
    # Given the 600s limit and that 2D DE + Polish is fast, we have plenty of time.
    # Let's try a few 12D DEs with a tight bound around the current best to see if we can escape.
    
    if time_ok() and best_roots is not None:
        # Define tight bounds around the current best log_roots
        # +/- 1.0 in log space is a factor of e ~ 2.7
        tight_bounds = [(x - 1.0, x + 1.0) for x in best_log_roots]
        
        seeds_12d = [111, 222, 333]
        for seed in seeds_12d:
            if not time_ok():
                break
            try:
                result_12d = differential_evolution(
                    cost_func, 
                    tight_bounds, 
                    seed=seed, 
                    maxiter=50, 
                    popsize=10, 
                    tol=1e-10, 
                    polish=True
                )
                
                if result_12d.fun < best_val:
                    best_val = result_12d.fun
                    best_roots = np.exp(result_12d.x)
                    
            except Exception:
                pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
