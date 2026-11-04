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

    # Precompute weights for norm calculation
    weights = np.arange(13) + 1  # (k+1) for k=0..12

    def compute_p_coeffs(roots):
        """Compute polynomial coefficients in ascending order."""
        coeffs_desc = np.poly(roots)
        return coeffs_desc[::-1]

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        p = compute_p_coeffs(roots)
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def compute_norm_sq_and_grad(roots, log_roots):
        """
        Compute norm_sq and its gradient with respect to log_roots.
        roots: array of 12 positive numbers.
        log_roots: array of 12 log-root values.
        Returns: (norm_sq, grad_log)
        """
        p = compute_p_coeffs(roots)
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0, np.zeros(12)
        
        # Gradient of norm_sq with respect to roots:
        # d(norm_sq)/d(roots_i) = 2 * c^T * A_T_inv * dp/d(roots_i)
        # where p = coeffs of prod(x - r_j)
        
        # dp/d(r_i) for each coefficient
        # p(x) = prod_j (x - r_j) = sum_k e_k * x^k where e_k are elementary symmetric polynomials
        # d(e_k)/d(r_i) = -e_{k-1}^{(i)} where e_{k-1}^{(i)} is elementary sym poly of all roots except r_i
        
        grad_roots = np.zeros(12)
        for i in range(12):
            other_roots = np.delete(roots, i)
            if len(other_roots) == 0:
                dp = np.zeros(13)
                dp[0] = 1.0
            else:
                coeffs_desc_i = np.poly(other_roots)
                dp = coeffs_desc_i[::-1]  # ascending
            # dp/d(r_i) = -dp (since d/d(r_i) prod(x-r_j) = -prod_{j!=i}(x-r_j))
            dp_dr_i = -dp
            # Extend to length 13
            if len(dp_dr_i) < 13:
                dp_dr_i = np.pad(dp_dr_i, (0, 13 - len(dp_dr_i)))
            
            # dc/d(roots_i) = A_T_inv @ dp_dr_i
            dc = A_T_inv @ dp_dr_i
            # d(norm_sq)/d(roots_i) = 2 * c^T * (weights * dc)
            grad_roots[i] = 2.0 * np.dot(c, weights * dc)
        
        # Chain rule: d(norm_sq)/d(log_roots_i) = d(norm_sq)/d(roots_i) * roots_i
        grad_log = grad_roots * roots
        
        return norm_sq, grad_log

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

    def cost_func_with_grad(log_roots):
        """
        Objective with analytical gradient for L-BFGS-B.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros(12)
            
        norm_sq, grad_log = compute_norm_sq_and_grad(roots, log_roots)
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(12)
        
        # d(1/norm_sq)/d(log_roots_i) = -1/norm_sq^2 * d(norm_sq)/d(log_roots_i)
        val = 1.0 / norm_sq
        grad = -grad_log / (norm_sq ** 2)
        
        return val, grad

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Multi-start local polish using L-BFGS-B with analytical gradients.
    # 3. Final high-precision polish pass.
    
    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = seed_log_roots.copy()

    # Run Differential Evolution
    seeds = [42, 123, 456, 789, 101]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=20, 
                tol=1e-8, 
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback / Initial guess
    seed_roots = np.exp(seed_log_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Phase 2: Multi-start local polish with analytical gradients
    # Generate diverse starting points for local optimization
    if time_ok():
        lb = -4.0
        ub = 10.0
        bounds_log = [(lb, ub)] * 12
        
        # Starting points: best from DE, seed, and random perturbations
        start_points = [best_log_roots.copy(), seed_log_roots.copy()]
        
        # Add some random starts
        rng = np.random.default_rng(12345)
        for _ in range(5):
            random_start = rng.uniform(lb, ub, size=12)
            start_points.append(random_start)
        
        # Also try starts based on different geometric sequences
        for q in [1.5, 2.5, 3.0]:
            for a in [2.0, 4.0, 8.0]:
                geo_start = np.log(a) + np.arange(12) * np.log(q)
                if np.all(geo_start >= lb) and np.all(geo_start <= ub):
                    start_points.append(geo_start)
        
        for sp in start_points:
            if not time_ok():
                break
            
            try:
                res = minimize(
                    cost_func_with_grad,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds_log,
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(best_log_roots)
                    
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if time_ok():
        try:
            lb = -4.0
            ub = 10.0
            bounds_log = [(lb, ub)] * 12
            
            res = minimize(
                cost_func_with_grad,
                best_log_roots,
                method='L-BFGS-B',
                bounds=bounds_log,
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
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
