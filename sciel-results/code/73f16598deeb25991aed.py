"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre
import time
from math import comb, factorial, log, exp

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    C4 <= K / sqrt(prod(roots_i))
    To minimize C4, we maximize prod(roots_i).
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    
    log_prod = np.sum(np.log(roots))
    c4_bound = exp(-0.5 * log_prod)
    return c4_bound

def _check_constraints(roots, tol=1e-6):
    """
    Check if the roots satisfy the Laguerre-type polynomial constraints.
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0):
        return False
    
    # Check that roots are distinct
    if np.any(np.diff(np.sort(roots)) < 1e-8):
        return False
    
    # Reasonable range check
    inv_sum = np.sum(1.0 / roots)
    if inv_sum < 0.1 or inv_sum > 100:
        return False
    
    return True

def _get_laguerre_zeros(n, alpha):
    """Get zeros of the generalized Laguerre polynomial L_n^(alpha)."""
    # Build the polynomial coefficients for L_n^(alpha)
    # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * C(n+alpha, n-k) * x^k / k!
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        coeffs[k] = ((-1)**k) * comb(n + alpha, n - k) / factorial(k)
    
    # coeffs[k] is the coefficient of x^k
    # For np.roots, we need highest degree first
    coeffs_desc = coeffs[::-1]
    
    roots = np.roots(coeffs_desc)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 1e-10]
    real_roots.sort()
    return real_roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin
    
    n = 12
    best_roots = None
    best_score = float('inf')
    
    # Strategy: Use generalized Laguerre polynomials with large alpha
    # to spread roots over a wider range, increasing the product of roots
    # and thus decreasing the 1/sqrt(product) bound.
    
    # Try various alpha values to find the one that maximizes the product of roots
    alpha_values = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    
    for alpha in alpha_values:
        if time.time() > deadline - 10:
            break
        
        try:
            roots = _get_laguerre_zeros(n, alpha)
            if len(roots) != 12:
                continue
            
            if not _check_constraints(roots):
                continue
            
            score = _compute_c4_bound(roots)
            if score < best_score:
                best_score = score
                best_roots = np.array(roots)
        except Exception:
            continue
    
    # If we haven't found good roots, fall back to standard Laguerre zeros
    if best_roots is None:
        try:
            roots = _get_laguerre_zeros(n, 0)
            if len(roots) == 12:
                best_roots = np.array(roots)
                best_score = _compute_c4_bound(roots)
        except Exception:
            best_roots = np.array([4.0 * (i + 1) for i in range(12)])
            best_score = _compute_c4_bound(best_roots)
    
    # Refine: try to optimize the roots while maintaining the Laguerre-like structure
    # The key insight is that for large alpha, the roots are well-spread
    # We can try to further optimize by adjusting individual roots
    
    if time.time() < deadline - 30:
        from scipy.optimize import minimize
        
        # Objective: minimize -log(product of roots) = -sum(log(roots))
        # In log-space: params = log(roots), objective = -sum(params)
        
        best_params = np.log(best_roots)
        
        # Use L-BFGS-B to refine
        def objective(params):
            return -np.sum(params)
        
        lb = np.full(12, -10.0)
        ub = np.full(12, 20.0)
        bounds = list(zip(lb, ub))
        
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < -np.sum(best_params):
                candidate_roots = np.exp(result.x)
                if np.all(candidate_roots > 0) and _check_constraints(candidate_roots):
                    score = _compute_c4_bound(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots
        except Exception:
            pass
    
    # Sort and return
    best_roots = np.sort(best_roots)
    return list(best_roots)

# EVOLVE-BLOCK-END
