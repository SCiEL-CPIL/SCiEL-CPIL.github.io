"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root locations."""
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.diff(np.sort(roots)) <= 0):
        return 1e10
    
    # Number of quadrature points for good accuracy
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Build the polynomial P(x) = prod (x - r_i)^2 * L_1^{(24)}(x)
    # L_1^{(alpha)}(x) = -(x) + (alpha + 1) = -x + 25
    laguerre_part = -x_quad + 25.0
    
    # Product of (x - r_i)^2
    prod_part = np.ones_like(x_quad)
    for r in roots:
        prod_part *= (x_quad - r) ** 2
    
    P_vals = prod_part * laguerre_part
    norm_sq = np.sum(w_quad * P_vals ** 2)
    
    if norm_sq <= 0:
        return 1e10
    
    return 1.0 / np.sqrt(norm_sq)

def _compute_c4_relative(roots):
    """Compute C4 bound relative to seed configuration for calibration."""
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    seed_val = _compute_c4_bound(seed_roots)
    if seed_val <= 0:
        return 1.0
    return _compute_c4_bound(roots) / seed_val

def _objective_free(params):
    """Objective: minimize C4 bound using free 12-dimensional parameterization.
    Parameters are log-spaced to ensure positivity and ordering."""
    # params[0] = log of first root
    # params[1:] = log of ratios between consecutive roots
    if len(params) != 12:
        return 1e10
    
    log_r0 = params[0]
    log_ratios = params[1:]
    
    r0 = np.exp(log_r0)
    ratios = np.exp(log_ratios)
    
    # Build roots from r0 and ratios
    roots = np.empty(12)
    roots[0] = r0
    for i in range(1, 12):
        roots[i] = roots[i-1] * ratios[i-1]
    
    # Check constraints
    if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    return _compute_c4_relative(roots)

def _objective_logspace(params):
    """Objective: minimize C4 bound using log-spaced parameterization.
    params[i] = log(roots[i]) with constraint that they're increasing."""
    if len(params) != 12:
        return 1e10
    
    roots = np.exp(params)
    
    # Check constraints
    if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    return _compute_c4_relative(roots)

def construct():
    """Construct 12 positive double-root locations minimizing C4."""
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 60.0  # seconds before deadline
    
    best_score = 1.0  # seed value
    best_roots = [4.0 * (i + 1) for i in range(12)]
    
    # Use log-space parameterization: optimize log(roots[i]) directly
    # This gives 12 free degrees of freedom
    
    # Initial guess: log of seed roots
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_log = np.log(seed_roots)
    
    # Bounds for log-space: roots in [0.1, 1000] => logs in [-2.3, 6.9]
    bounds = [(-2.3, 6.9)] * 12
    
    np.random.seed(42)
    
    # Multistart with various perturbations
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            if i == 0:
                x0 = seed_log.copy()
            else:
                # Perturb the seed
                scale = np.random.uniform(0.5, 2.0)
                x0 = seed_log + np.random.normal(0, 0.5, size=12) * scale
            
            result = minimize(
                _objective_logspace,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    best_score = result.fun
                    best_roots = list(roots)
                    
        except Exception:
            continue
    
    # If we found something better, return it; otherwise return seed
    if best_score < 1.0:
        return best_roots
    else:
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
