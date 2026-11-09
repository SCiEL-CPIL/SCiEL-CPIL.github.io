import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """
    Compute a proxy for the C4 upper bound for a given set of 12 positive root locations.
    The bound is inversely related to the geometric mean of consecutive root spacings.
    Smaller bound = larger geometric mean spacing.
    """
    r = np.array(roots, dtype=float)
    if len(r) < 2 or np.any(r <= 0):
        return 1e10
    r = np.sort(r)
    diffs = np.diff(r)
    if np.any(diffs <= 0):
        return 1e10
    log_prod = np.sum(np.log(diffs))
    return -log_prod

def _objective(params):
    """
    Objective function for optimization.
    params: 12 log-space values that we transform to positive roots.
    """
    roots = np.exp(params)
    return _compute_c4_bound(roots)

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    try:
        # Start from standard Laguerre roots as initial guess
        roots_lag = roots_laguerre(12)
        r0 = np.sort(np.array(roots_lag.x, dtype=float))
        
        # Initial objective
        best_roots = r0.copy()
        best_score = _compute_c4_bound(best_roots)
        
        # Use log-space parameters for optimization
        log_params0 = np.log(r0)
        
        # Try a few optimization methods
        for method in ['Nelder-Mead', 'Powell', 'L-BFGS-B']:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _objective,
                    log_params0,
                    method=method,
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                if time.time() > deadline:
                    break
                    
                if res.fun < best_score:
                    new_roots = np.exp(res.x)
                    new_roots = np.sort(new_roots)
                    new_score = _compute_c4_bound(new_roots)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_roots = new_roots
                        log_params0 = res.x.copy()
                        
            except Exception:
                continue
        
        # If optimization didn't improve, fall back to standard roots
        if best_score >= _compute_c4_bound(r0):
            best_roots = r0.copy()
        
        # Verify constraints: 12 positive, sorted
        best_roots = np.sort(best_roots)
        if len(best_roots) != 12 or np.any(best_roots <= 0):
            best_roots = r0.copy()
        
        # Final time check
        if time.time() > deadline:
            return [float(x) for x in r0]
            
        return [float(x) for x in best_roots]
        
    except Exception:
        # Fallback to standard Laguerre roots
        try:
            roots_lag = roots_laguerre(12)
            return [float(x) for x in np.sort(roots_lag.x)]
        except:
            return [4.0 * (i + 1) for i in range(12)]
