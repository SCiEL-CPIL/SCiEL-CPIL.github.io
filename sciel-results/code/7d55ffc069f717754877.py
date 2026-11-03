import time
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize
from scipy.special import eval_laguerre, gammaln, gamma
import math

_START = time.time()
_DEADLINE = _START + 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - time.time()

def _compute_c4_bound(roots, m=12, n_dim=1):
    """
    Compute the upper bound on C4 from the Laguerre polynomial construction.
    
    The construction: P(x) = prod_{i=1}^{m} (x - r_i)^2
    The norm is computed via the Laguerre inner product:
        ||P||^2 = integral_0^inf P(x)^2 * x^{alpha} * exp(-x) dx
    where alpha = 2 * n_dim = 2.
    
    The upper bound on C4 is proportional to 1/sqrt(||P||^2).
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e9
    if np.any(roots <= 0):
        return 1e9
    if np.any(np.diff(roots) <= 0):
        return 1e9
    
    alpha = 2 * n_dim  # = 2
    
    # Build Q(x) = prod_{i=0}^{m-1} (x - r_i)
    q_coeff = np.array([1.0])
    for r in roots:
        q_coeff = np.convolve(q_coeff, np.array([1.0, -r]))
    
    # P(x) = Q(x)^2
    p_coeff = np.convolve(q_coeff, q_coeff)
    
    # ||P||^2 = sum_{i,j} p_coeff[i] * p_coeff[j] * Gamma(i + j + alpha + 1)
    indices = np.arange(len(p_coeff))
    gamma_args = indices[:, None] + indices[None, :] + alpha + 1
    log_gamma_vals = np.log(gamma(gamma_args))
    log_gamma_vals = np.clip(log_gamma_vals, -700, 700)
    gamma_vals = np.exp(log_gamma_vals)
    
    norm_sq = p_coeff @ (gamma_vals @ p_coeff)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    return 1.0 / np.sqrt(norm_sq)

def _objective_geometric(params):
    """Objective in log-geometric parameterization: roots = r0 * q^i"""
    log_r0, log_q = params
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    
    if r0 <= 0 or q <= 0:
        return 1e10
    
    roots = r0 * (q ** np.arange(12))
    
    if np.any(roots <= 0) or np.any(roots > 1e10) or np.any(roots < 1e-10):
        return 1e10
    
    return _compute_c4_bound(roots)

def _construct():
    m = 12
    n_dim = 1
    best = None
    best_val = 1e18
    
    # Phase 1: Start from Laguerre nodes
    try:
        x0, _ = laguerre.laggauss(m)
        x0 = np.sort(x0)
        best = x0.copy()
        best_val = _compute_c4_bound(best, m, n_dim)
    except Exception:
        best = np.linspace(0.5, 20.0, m)
        best_val = _compute_c4_bound(best, m, n_dim)
    
    # Phase 2: Geometric-parameterized multistart optimization
    if _time_left() > 60.0:
        np.random.seed(42)
        
        start_points = [
            (np.log(4.0), np.log(12 ** (1.0 / 11))),
            (np.log(2.0), np.log(2.0)),
            (np.log(1.0), np.log(1.5)),
            (np.log(5.0), np.log(1.3)),
            (np.log(3.0), np.log(1.8)),
            (np.log(1.0), np.log(2.0)),
            (np.log(10.0), np.log(1.2)),
            (np.log(0.5), np.log(2.5)),
        ]
        
        for _ in range(10):
            lr0 = np.random.uniform(np.log(0.1), np.log(20.0))
            lq = np.random.uniform(np.log(0.5), np.log(4.0))
            start_points.append((lr0, lq))
        
        for sp in start_points:
            if _time_left() < 15.0:
                break
            try:
                result = minimize(
                    _objective_geometric,
                    x0=np.array(sp),
                    method='L-BFGS-B',
                    bounds=[(np.log(0.01), np.log(100.0)), (np.log(0.1), np.log(10.0))],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_val:
                    log_r0, log_q = result.x
                    r0 = np.exp(log_r0)
                    q = np.exp(log_q)
                    roots = r0 * (q ** np.arange(12))
                    
                    if np.all(roots > 0) and np.all(np.isfinite(roots)):
                        score = _compute_c4_bound(roots, m, n_dim)
                        if score < best_val:
                            best_val = score
                            best = roots
            except Exception:
                continue
    
    # Phase 3: Full 12-dimensional local refinement
    if _time_left() > 30.0:
        try:
            def objective(roots):
                roots = np.sort(roots)
                if np.any(roots <= 1e-8):
                    return 1e9
                if np.any(np.diff(roots) <= 1e-10):
                    return 1e9
                return _compute_c4_bound(roots, m, n_dim)
            
            res = minimize(
                objective, best,
                method='L-BFGS-B',
                bounds=[(1e-8, 100.0)] * m,
                options={'maxiter': 3000, 'maxfun': 50000, 'ftol': 1e-12},
            )
            if res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    # Phase 4: Multistart full-dimensional optimization
    if _time_left() > 60.0:
        inits = []
        
        # Geometric spacing
        inits.append(np.geomspace(0.5, 25.0, m))
        
        # Linear spacing
        inits.append(np.linspace(1.0, 20.0, m))
        
        # Clustered near origin
        inits.append(np.sort(np.random.uniform(0.5, 10.0, m)))
        
        # Spread out
        inits.append(np.sort(np.random.uniform(1.0, 50.0, m)))
        
        # Fit geometric to current best
        if best is not None:
            logs = np.log(best)
            A = np.vstack([np.ones(m), np.arange(m)]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, logs, rcond=None)
                r0_fit = np.exp(coeffs[0])
                q_fit = np.exp(coeffs[1])
                if 0.1 < q_fit < 10 and r0_fit > 0:
                    inits.append(r0_fit * (q_fit ** np.arange(m)))
            except Exception:
                pass
        
        for init in inits:
            if _time_left() < 15.0:
                break
            try:
                res = minimize(
                    objective, init,
                    method='L-BFGS-B',
                    bounds=[(1e-8, 100.0)] * m,
                    options={'maxiter': 2000, 'maxfun': 30000, 'ftol': 1e-12},
                )
                if res.fun < best_val:
                    cand = np.sort(res.x)
                    if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                        best_val = res.fun
                        best = cand
            except Exception:
                pass
    
    # Phase 5: Final polish
    if _time_left() > 10.0:
        try:
            res = minimize(
                objective, best,
                method='L-BFGS-B',
                bounds=[(1e-8, 100.0)] * m,
                options={'maxiter': 5000, 'maxfun': 100000, 'ftol': 1e-14},
            )
            if res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    best = np.sort(np.maximum(best, 1e-8))
    return [float(v) for v in best]

def construct():
    """Return 12 positive double-root locations for the refined
    Cohn-Goncalves Laguerre construction (m=12, n=1)."""
    try:
        return _construct()
    except Exception:
        x, _ = laguerre.laggauss(12)
        return [float(v) for v in x]
