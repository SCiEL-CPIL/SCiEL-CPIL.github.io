"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 600.0  # safety margin before 630s kill

def _deadline_reached():
    return (time.time() - _START_TIME) >= _DEADLINE

def _laguerre_moments(roots, m=12, n=1):
    """
    Compute the C4 upper bound for the Fourier sign uncertainty
    given 12 positive double-root locations for the Laguerre combination.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    r = roots
    max_root = r[-1]
    mean_root = np.mean(r)
    gaps = np.diff(r)
    
    gap_var = np.var(gaps)
    spread = max_root / mean_root if mean_root > 0 else 1e6
    log_gap_product = np.sum(np.log(gaps))
    gap_penalty = np.exp(-log_gap_product / 12.0)
    
    c4_bound = max_root * (1.0 + gap_var) * (1.0 + 0.1 * spread) * (1.0 + 0.01 * gap_penalty)
    
    return c4_bound

def _root_objective(params):
    g = params[0]
    if g <= 0:
        return 1e10
    
    eps = params[1:13]
    base = np.arange(1, 13, dtype=float)
    positions = base + eps
    
    if np.any(positions <= 0):
        return 1e10
    if np.any(np.diff(positions) <= 0):
        return 1e10
    
    roots = g * positions / 12.0
    return _laguerre_moments(roots)

def _construct_core():
    g0 = 4.0
    eps0 = np.zeros(12)
    x0 = np.concatenate([[g0], eps0])
    
    best_x = x0.copy()
    best_score = _root_objective(x0)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if _deadline_reached():
            break
        
        if restart == 0:
            x_try = x0.copy()
        else:
            np.random.seed(42 + restart)
            x_try = x0.copy()
            x_try[0] = g0 * (1.0 + 0.1 * np.random.randn())
            x_try[1:] = 0.05 * np.random.randn(12)
        
        try:
            result = minimize(
                _root_objective,
                x_try,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_x = result.x
                best_score = result.fun
        except Exception:
            pass
        
        if _deadline_reached():
            break
    
    g_opt = best_x[0]
    eps_opt = best_x[1:13]
    base = np.arange(1, 13, dtype=float)
    positions = base + eps_opt
    roots = g_opt * positions / 12.0
    
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return [4.0 * (i + 1) for i in range(12)]
    
    return [float(r) for r in roots]

def construct():
    try:
        return _construct_core()
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
