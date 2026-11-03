"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _compute_log_norm(roots):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)^2
    with weight e^{-x}.
    We want to MAXIMIZE this norm.
    """
    m = len(roots)
    # Use Gauss-Laguerre quadrature
    # Roots are typically in the range [1, 100]. 300 nodes is safe for m=12.
    n_quad = 300
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log P(x)^2 = 4 * sum log|x - r_i|
    # Note: P(x) = prod (x-r_i)^2, so |P(x)|^2 = prod (x-r_i)^4
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        # Avoid log(0)
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf
    
    sum_val = np.sum(w_quad * np.exp(log_integrand - max_log))
    
    if sum_val <= 0:
        return -np.inf
        
    log_norm_sq = max_log + np.log(sum_val)
    
    return log_norm_sq

def _roots_from_geometric(mu, sigma):
    """Generate 12 roots from log-normal geometric progression."""
    i = np.arange(12)
    return np.exp(mu + sigma * i)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess based on previous knowledge: roots around 4, 8, ..., 48
    # Geometric mean of 4 and 48 is sqrt(192) ~ 13.8
    # 48/4 = 12, 12^(1/11) ~ 1.26
    # So mu ~ log(4) ~ 1.38, sigma ~ log(1.26) ~ 0.23
    # Let's center the search around these values.
    
    best_roots = None
    best_log_norm = -np.inf
    
    def objective_geometric(params):
        """Objective function for geometric parameterization (minimize negative log norm)."""
        if time.time() > deadline - 10:
            return 1e10
        mu, sigma = params
        # Ensure roots are positive and reasonably bounded
        # sigma can be negative, but roots must be positive.
        # If sigma is too large, roots might be too spread out.
        roots = _roots_from_geometric(mu, sigma)
        if not np.all(np.isfinite(roots)) or np.any(roots <= 0) or np.any(roots > 1000):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm

    # Phase 1: Global search over (mu, sigma)
    # Bounds:
    # mu: log(root_min). Roots should be > 1. log(1)=0. Let's say [0, 5]
    # sigma: log(ratio). Ratio between adjacent roots. 
    # If roots are 4..48, ratio is ~1.26, log~0.23. 
    # Could be tighter or wider. Let's say [-0.5, 1.0]
    bounds = [(0.0, 5.0), (-0.5, 1.0)]
    
    try:
        # Differential Evolution is a global optimizer
        de_result = differential_evolution(
            objective_geometric, 
            bounds, 
            seed=42, 
            maxiter=500, 
            tol=1e-12,
            popsize=15,
            init='sobol',
            polish=False # We will do our own polish
        )
        
        if de_result.success or de_result.fun < 1e9:
            mu_opt, sigma_opt = de_result.x
            best_roots = _roots_from_geometric(mu_opt, sigma_opt)
            best_log_norm = -de_result.fun
            
    except Exception:
        pass
    
    # Phase 2: Local refinement on 12 independent roots
    if best_roots is None:
        # Fallback seed
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        best_log_norm = _compute_norm_log(best_roots) if '_compute_norm_log' in dir() else _compute_log_norm(best_roots)
    else:
        best_log_norm = _compute_log_norm(best_roots)
        
    def objective_full(params):
        """Objective function for full 12-dimensional root optimization."""
        if time.time() > deadline - 5:
            return 1e10
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 1000):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm

    try:
        # Refine with L-BFGS-B
        result = minimize(
            objective_full, 
            best_roots, 
            method='L-BFGS-B',
            bounds=[(1e-6, 1000)] * 12,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = -result.fun
            
    except Exception:
        pass
        
    # Final check
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        # Fallback to seed
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
