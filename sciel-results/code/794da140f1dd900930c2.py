import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_norm_log_quartic(roots, n_quad=400):
    """
    Compute the log of the L2 norm squared of P(x)^2 = prod (x - r_i)^2
    with weight e^{-x}.
    This is equivalent to computing log( integral P(x)^4 e^{-x} dx ).
    """
    m = len(roots)
    # Use more quadrature points for higher accuracy
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log |P(x)|^4 = 4 * sum log|x - r_i|
    log_p4 = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        # Avoid log(0)
        diff[diff < 1e-300] = 1e-300
        log_p4 += 4.0 * np.log(diff)
    
    log_integrand = -x_quad + log_p4
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf
    
    sum_val = np.sum(w_quad * np.exp(log_integrand - max_log))
    if sum_val <= 0:
        return -np.inf
        
    log_norm_sq = max_log + np.log(sum_val)
    return log_norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: Roots of L_12(x)
    lag_roots, _ = roots_laguerre(12)
    best_roots = np.sort(lag_roots)
    best_score = _compute_norm_log_quartic(best_roots)
    
    # Also try roots of L_12^{(1)}(x) just in case
    try:
        # scipy.special.roots_genlaguerre(n, alpha)
        from scipy.special import roots_genlaguerre
        lag_roots_1, _ = roots_genlaguerre(12, 1.0)
        cand = np.sort(lag_roots_1)
        score = _compute_norm_log_quartic(cand)
        if score > best_score:
            best_score = score
            best_roots = cand
    except:
        pass

    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return 1e10
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10
        log_norm = _compute_norm_log_quartic(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm

    # Local refinement
    for start in [best_roots]:
        if time.time() - start_time > deadline - 30:
            break
        try:
            result = minimize(objective, start, method='L-BFGS-B',
                             bounds=[(1e-6, None)] * 12,
                             options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14})
            if np.isfinite(result.fun) and -result.fun > best_score:
                best_score = -result.fun
                best_roots = np.sort(result.x)
        except Exception:
            pass

    # Global search if time permits
    if time.time() - start_time < deadline - 120:
        from scipy.optimize import differential_evolution
        def de_objective(params):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                return 1e10
            roots = np.sort(params)
            if any(r <= 1e-6 for r in roots):
                return 1e10
            log_norm = _compute_norm_log_quartic(roots)
            if not np.isfinite(log_norm):
                return 1e10
            return -log_norm
        
        # Bounds: Laguerre roots are roughly in [0, 4m]
        bounds_de = [(0.1, 60.0)] * 12
        try:
            result_de = differential_evolution(
                de_objective, bounds_de, seed=42, maxiter=50,
                popsize=15, tol=1e-8, polish=True,
                options={'maxiter': 50, 'init': 'sobol'}
            )
            if np.isfinite(result_de.fun) and -result_de.fun > best_score:
                best_score = -result_de.fun
                best_roots = np.sort(result_de.x)
        except Exception:
            pass

    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = np.sort(roots_laguerre(12)[0])
        
    return best_roots.tolist()
