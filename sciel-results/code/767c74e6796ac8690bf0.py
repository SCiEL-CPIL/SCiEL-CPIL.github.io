"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, gammainc
import time
from math import comb, factorial, log, exp

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    In the Cohn-Goncalves construction, C4 is related to the product of roots
    and the Laguerre polynomial evaluation.
    
    For m=12 double roots, the bound involves:
    C4 <= K / sqrt(prod(roots_i))
    where K is a constant depending on the construction.
    
    To minimize C4, we maximize prod(roots_i).
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    
    # The C4 bound for the refined construction is proportional to
    # 1 / sqrt(prod(roots))
    # So minimizing C4 = maximizing prod(roots)
    log_prod = np.sum(np.log(roots))
    c4_bound = exp(-0.5 * log_prod)
    return c4_bound

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 590.0  # Safety margin for 600s limit
    
    # Start with zeros of L_12(x) as initial guess
    n = 12
    
    # Calculate coefficients of L_n(x)
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        if k == 0:
            coeffs[0] = 1.0
        else:
            coeffs[k] = ((-1)**k) * comb(n + k, k) / factorial(k)
    
    coeffs_desc = coeffs[::-1]
    
    # Find roots of L_12(x)
    try:
        roots_lag = np.roots(coeffs_desc)
        real_roots = [r.real for r in roots_lag if abs(r.imag) < 1e-8 and r.real > 1e-10]
        real_roots.sort()
        
        if len(real_roots) != 12:
            raise ValueError("Wrong number of real roots")
        
        x0 = np.array(real_roots)
    except Exception:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
    
    # Now try to optimize: maximize product of roots (minimize C4 bound)
    # subject to the constraint that these are valid double roots for the construction.
    
    # The key constraint: for the Cohn-Goncalves construction with m=12, n=1,
    # the roots must satisfy that the Laguerre-type polynomial with these double roots
    # has the correct sign change properties.
    
    # A simpler structural change: use the fact that for optimal C4,
    # the roots should be scaled. Let's try scaling the Laguerre zeros.
    
    # Try different scaling factors to see which gives better C4 bound
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Try scaling
    for scale in np.linspace(0.5, 2.0, 50):
        candidate = x0 * scale
        if np.all(candidate > 0):
            score = _compute_c4_bound(candidate)
            if score < best_score:
                best_score = score
                best_roots = candidate.copy()
    
    # Try shifting
    for shift in np.linspace(-2.0, 2.0, 50):
        candidate = x0 + shift
        if np.all(candidate > 0):
            score = _compute_c4_bound(candidate)
            if score < best_score:
                best_score = score
                best_roots = candidate.copy()
    
    # Try combination of scale and shift
    for scale in np.linspace(0.5, 2.0, 20):
        for shift in np.linspace(-2.0, 2.0, 20):
            candidate = x0 * scale + shift
            if np.all(candidate > 0):
                score = _compute_c4_bound(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return list(best_roots)
    
    # Try local optimization using scipy
    from scipy.optimize import minimize
    
    def objective(params):
        roots = np.exp(params)  # Ensure positivity via log transform
        return -np.sum(np.log(roots))  # Maximize product = minimize negative log product
    
    # Initial guess in log space
    x0_log = np.log(x0)
    
    try:
        result = minimize(objective, x0_log, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.success:
            optimized_roots = np.exp(result.x)
            if np.all(optimized_roots > 0):
                score = _compute_c4_bound(optimized_roots)
                if score < best_score:
                    best_score = score
                    best_roots = optimized_roots.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        return list(best_roots)
    
    # Sort the final roots
    best_roots = np.sort(best_roots)
    return list(best_roots)

# EVOLVE-BLOCK-END
