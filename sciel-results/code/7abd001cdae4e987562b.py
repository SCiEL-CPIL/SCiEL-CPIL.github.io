"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv_max(h):
    """Compute supremum norm of autoconvolution."""
    # Full autoconvolution
    c = np.correlate(h, h, mode='full')
    return np.max(c)

def _ratio(h):
    """Compute the objective ratio."""
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    max_conv = _autoconv_max(h)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _neg_ratio(h):
    return -_ratio(h)

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Start with a single spike (known good: ratio=1)
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _ratio(best_h)
    
    # Try a few initial patterns
    patterns = []
    # Single spike at center
    p = np.zeros(N); p[N//2] = 1.0; patterns.append(p)
    # Two spikes
    p = np.zeros(N); p[0] = 1.0; p[N//2] = 1.0; patterns.append(p)
    # Triangular
    p = np.linspace(1, 0, N//2); p = np.concatenate([p, p[::-1][:N//2]]); patterns.append(p)
    # Sparse random
    np.random.seed(42)
    for trial in range(5):
        p = np.zeros(N)
        idx = np.random.choice(N, size=5, replace=False)
        p[idx] = 1.0
        patterns.append(p)
    
    for p in patterns:
        v = _ratio(p)
        if v > best_val:
            best_val = v
            best_h = p.copy()
    
    # Local optimization from best starting point
    if time.time() < deadline:
        x0 = best_h.copy()
        result = minimize(_neg_ratio, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < -best_val:
            best_val = -result.fun
            best_h = np.maximum(result.x, 0)  # enforce nonnegativity
    
    # Try optimization from a few other starts
    for p in patterns[:3]:
        if time.time() >= deadline - 20:
            break
        x0 = p.copy()
        result = minimize(_neg_ratio, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < -best_val:
            best_val = -result.fun
            best_h = np.maximum(result.x, 0)
    
    # Polish: try SLSQP with bounds
    if time.time() < deadline - 10:
        bounds = [(0, None)] * N
        result = minimize(_neg_ratio, best_h, method='SLSQP', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-12})
        if result.fun < -best_val:
            best_val = -result.fun
            best_h = np.maximum(result.x, 0)
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        result = minimize(_neg_ratio, best_h, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < -best_val:
            best_val = -result.fun
            best_h = np.maximum(result.x, 0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
