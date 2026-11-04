import time
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_laguerre, gammaln, gamma
import math

_START = time.time()
_DEADLINE = _START + 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - time.time()

def _compute_c4(roots, m=12):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    
    Given m=12 double roots, the C4 bound is computed from the
    Laguerre-weighted moments of the polynomial P(x) = prod (x-r_i)^2.
    
    C4 is the standard deviation of x under the measure P(x)^2 * e^{-x} dx,
    which quantifies the effective spread of the sign uncertainty.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e9
    if np.any(roots <= 0):
        return 1e9
    if np.any(np.diff(roots) <= 0):
        return 1e9
    
    # Gauss-Laguerre quadrature for moments
    x_q, w_q = laguerre.laggauss(50)
    
    # log P(x)^2 = 4 * sum log|x - r_i|
    log_P2 = 4.0 * np.sum(np.log(np.abs(x_q[:, None] - roots[None, :])), axis=1)
    log_P2 = np.clip(log_P2, -400, 400)
    P2 = np.exp(log_P2)
    
    m0 = np.sum(w_q * P2)
    m1 = np.sum(w_q * P2 * x_q)
    m2 = np.sum(w_q * P2 * x_q**2)
    
    if m0 <= 1e-300:
        return 1e9
    
    mean = m1 / m0
    var = m2 / m0 - mean**2
    c4 = math.sqrt(max(var, 0.0))
    
    return float(c4)

def _construct():
    m = 12
    best = None
    best_val = 1e18
    
    # Phase 1: Start from tightly packed roots near origin
    # Since C4 ~ std(x) under the measure, we want roots clustered
    best = np.linspace(0.1, 5.0, m)
    best_val = _compute_c4(best, m)
    
    # Phase 2: Local refinement using L-BFGS-B in log-space
    if _time_left() > 30.0:
        try:
            def objective_log(log_roots):
                roots = np.exp(log_roots)
                roots = np.sort(roots)
                if np.any(roots <= 1e-10):
                    return 1e9
                if np.any(np.diff(roots) <= 1e-12):
                    return 1e9
                return _compute_c4(roots, m)
            
            log_best = np.log(best)
            res = minimize(
                objective_log, log_best,
                method='L-BFGS-B',
                bounds=[(-10.0, 5.0)] * m,
                options={'maxiter': 5000, 'maxfun': 80000, 'ftol': 1e-14},
            )
            if res.fun < best_val:
                cand = np.sort(np.exp(res.x))
                if np.all(cand > 1e-10) and np.all(np.diff(cand) > 1e-12):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global exploration
    if _time_left() > 120.0:
        try:
            def de_objective(log_roots):
                roots = np.exp(log_roots)
                roots = np.sort(roots)
                if np.any(roots <= 1e-10):
                    return 1e9
                if np.any(np.diff(roots) <= 1e-12):
                    return 1e9
                return _compute_c4(roots, m)
            
            bounds_de = [(-8.0, 3.0)] * m
            res_de = differential_evolution(
                de_objective, bounds_de,
                seed=42, maxiter=200, popsize=20,
                tol=1e-10, polish=True,
                update_interval=10,
                callback=None,
                init='latinhypercube',
            )
            if res_de.fun < best_val:
                cand = np.sort(np.exp(res_de.x))
                if np.all(cand > 1e-10) and np.all(np.diff(cand) > 1e-12):
                    best_val = res_de.fun
                    best = cand
        except Exception:
            pass
    
    # Phase 4: Final polish
    if _time_left() > 30.0:
        try:
            def objective_log2(log_roots):
                roots = np.exp(log_roots)
                roots = np.sort(roots)
                if np.any(roots <= 1e-10):
                    return 1e9
                if np.any(np.diff(roots) <= 1e-12):
                    return 1e9
                return _compute_c4(roots, m)
            
            log_best = np.log(best)
            res = minimize(
                objective_log2, log_best,
                method='L-BFGS-B',
                bounds=[(-10.0, 5.0)] * m,
                options={'maxiter': 10000, 'maxfun': 200000, 'ftol': 1e-16},
            )
            if res.fun < best_val:
                cand = np.sort(np.exp(res.x))
                if np.all(cand > 1e-10) and np.all(np.diff(cand) > 1e-12):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    best = np.sort(np.maximum(best, 1e-10))
    return [float(v) for v in best]

def construct():
    """Return 12 positive double-root locations for the refined
    Cohn-Goncalves Laguerre construction (m=12, n=1)."""
    try:
        return _construct()
    except Exception:
        # Safe fallback: tightly packed positive roots
        return [float(v) for v in np.linspace(0.1, 5.0, 12)]
