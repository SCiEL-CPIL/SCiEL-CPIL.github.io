# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Laguerre quadrature nodes and weights
_N_QUAD = 500
_x_quad, _w_quad = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_log_norm_and_grad_log(roots):
    """
    Compute the log of the L2 norm squared of P(x) = prod (x - r_i)^2
    with weight e^{-x}, and its gradient with respect to log(roots).
    We want to MAXIMIZE this norm.
    """
    roots = np.asarray(roots, dtype=np.float64)
    n_roots = len(roots)
    
    # Compute log P(x)^2 = 4 * sum log|x - r_i|
    log_p_sq = np.zeros(_N_QUAD)
    for r in roots:
        diff = np.abs(_x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -_x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros(n_roots)
    
    # Weighted contributions
    weights_exp = _w_quad * np.exp(log_integrand - max_log)
    sum_val = np.sum(weights_exp)
    
    if sum_val <= 0:
        return -np.inf, np.zeros(n_roots)
    
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient with respect to r_j:
    # d log_norm_sq / d r_j = -4 * sum(weights_exp / (x_i - r_j)) / sum_val
    # Gradient with respect to log_r_j:
    # d log_norm_sq / d log_r_j = r_j * (d log_norm_sq / d r_j)
    
    grad_log = np.zeros(n_roots)
    for j, r in enumerate(roots):
        diff = _x_quad - r
        # Avoid division by zero
        diff[diff < 1e-300] = 1e-300
        diff[diff > -1e-300] = -1e-300
        # d log_norm_sq / d r_j
        grad_r = -4.0 * np.sum(weights_exp / diff) / sum_val
        # d log_norm_sq / d log_r_j
        grad_log[j] = r * grad_r
        
    return log_norm_sq, grad_log

def _compute_log_norm(roots):
    """Compute the log of the L2 norm squared."""
    val, _ = _compute_log_norm_and_grad_log(roots)
    return val

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_roots = None
    best_log_norm = -np.inf
    
    # We optimize in log-space.
    # Initial guess: geometric progression
    # r_i = exp(mu + sigma * i)
    # log_r_i = mu + sigma * i
    
    def objective_log(params):
        """Objective function for log-roots (minimize negative log norm)."""
        if time.time() > deadline - 30:
            return 1e10
        log_roots = params
        roots = np.exp(log_roots)
        # Check for overflow or invalid roots
        if not np.all(np.isfinite(roots)) or np.any(roots <= 0) or np.any(roots > 1e6):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm

    def grad_log(params):
        """Gradient for log-roots (minimize negative log norm)."""
        if time.time() > deadline - 10:
            return np.zeros(12)
        log_roots = params
        roots = np.exp(log_roots)
        if not np.all(np.isfinite(roots)) or np.any(roots <= 0) or np.any(roots > 1e6):
            return np.zeros(12)
        _, grad = _compute_log_norm_and_grad_log(roots)
        # We're minimizing -log_norm, so gradient is -grad
        return -grad

    # Phase 1: Global search in log-space
    # Bounds for log_roots:
    # Roots should be between 0.1 and 1000.
    # log(0.1) ~ -2.3, log(1000) ~ 6.9
    bounds_log = [(-2.0, 7.0)] * 12
    
    try:
        de_result = differential_evolution(
            objective_log, 
            bounds_log, 
            seed=42, 
            maxiter=200, 
            tol=1e-10,
            popsize=15,
            init='sobol',
            polish=False
        )
        
        if de_result.success or de_result.fun < 1e9:
            best_log_roots = de_result.x
            best_roots = np.exp(best_log_roots)
            best_log_norm = _compute_log_norm(best_roots)
            
    except Exception:
        best_log_roots = None
    
    # Phase 2: Local refinement in log-space
    if best_log_roots is not None and time.time() < deadline - 60:
        try:
            result = minimize(
                objective_log, 
                best_log_roots, 
                jac=grad_log,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            
            if result.fun < 1e9 and result.fun < -best_log_norm:
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                best_log_norm = _compute_log_norm(best_roots)
                
        except Exception:
            pass

    # Phase 3: Multi-start local refinement if Phase 1 didn't yield a good result or to explore more
    # Generate diverse starting points in log-space
    start_points_log = []
    if best_log_roots is not None:
        start_points_log.append(best_log_roots.copy())
    
    # Geometric sequences in log-space are linear
    for mu_try in [0.5, 1.0, 1.5, 2.0]:
        for sigma_try in [0.1, 0.2, 0.3, 0.4]:
            log_rp = mu_try + sigma_try * np.arange(12)
            # Ensure within bounds
            if np.all(log_rp > -2.0) and np.all(log_rp < 7.0):
                start_points_log.append(log_rp)
                
    # Arithmetic in log-space (geometric in linear)
    # Already covered above.
    # Try some random perturbations of the best
    if best_log_roots is not None:
        for _ in range(5):
            perturbed = best_log_roots + np.random.normal(0, 0.1, 12)
            if np.all(perturbed > -2.0) and np.all(perturbed < 7.0):
                start_points_log.append(perturbed)

    for sp in start_points_log:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                objective_log, 
                sp, 
                jac=grad_log,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            
            if result.fun < 1e9 and result.fun < -best_log_norm:
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                best_log_norm = _compute_log_norm(best_roots)
                
        except Exception:
            pass

    # Final check
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
