"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0
    
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
    # p is vector of coeffs in standard basis (ascending)
    # c is vector of coeffs in Laguerre basis
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

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Local polish using L-BFGS-B.
    
    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    # log_r_i = log(4) + i * log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    # Roots should be positive.
    # If roots are too small or too large, the norm might behave badly.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = seed_log_roots.copy()

    # Run Differential Evolution
    # We have ~600s. DE is stochastic.
    # Let's run a few independent DEs with different seeds and settings.
    
    seeds = [42, 123, 456, 789, 101]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # popsize=15, maxiter=100 is usually fast for 12D.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=20, 
                tol=1e-8, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed from the prompt: r0=4, q=2
    seed_roots = np.exp(seed_log_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Phase 2: High-precision polish in log-space
    if time_ok():
        try:
            # L-BFGS-B on log_roots
            # Bounds for log_roots
            lb = -4.0
            ub = 10.0
            bounds_log = [(lb, ub)] * 12
            
            # Initial point
            x0 = best_log_roots
            
            # Objective for polish
            def obj_log(x):
                return cost_func(x)
            
            # Gradient is not provided, L-BFGS-B will use numerical differentiation
            # This is fine for 12D
            
            res = minimize(
                obj_log,
                x0,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
