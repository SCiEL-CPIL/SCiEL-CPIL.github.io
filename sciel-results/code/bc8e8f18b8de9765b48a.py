"""Second autocorrelation inequality - optimized."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_params(params):
    """params: raw values, h = exp(params) normalized to sum=1 (positivity guaranteed)"""
    e = np.exp(params - np.max(params))
    h = e / e.sum()
    return h

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    def try_optimize(params0, maxiter, xatol, fatol):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            res = minimize(_neg_value, params0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                                   'adaptive': True})
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = _h_from_params(res.x)
        except Exception:
            pass

    def try_bfgs(params0, maxiter):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            res = minimize(_neg_value, params0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-6})
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = _h_from_params(res.x)
        except Exception:
            pass

    rng = np.random.default_rng(50)
    x = np.linspace(-1, 1, N)

    # Diverse initial shapes
    candidates = []
    candidates.append(np.ones(N))
    candidates.append(np.exp(-2 * x**2))
    candidates.append(np.exp(-8 * x**2))
    candidates.append(np.exp(-0.5 * x**2))
    candidates.append(np.maximum(0, 1 - 2 * np.abs(np.linspace(-0.5, 0.5, N))))
    candidates.append(np.maximum(0, np.cos(np.pi * np.linspace(-0.5, 0.5, N))))
    candidates.append(np.maximum(0, np.cos(2 * np.pi * np.linspace(-0.5, 0.5, N))))
    candidates.append(np.maximum(0, 1 - np.abs(np.linspace(-1, 1, N))**4))
    candidates.append(np.maximum(0, np.sin(np.pi * np.linspace(-0.5, 0.5, N))))
    # Bimodal
    candidates.append(np.exp(-8*(x-0.4)**2) + np.exp(-8*(x+0.4)**2))
    candidates.append(np.exp(-12*(x-0.3)**2) + np.exp(-12*(x+0.3)**2))
    # Plateau
    plateau = np.zeros(N)
    plateau[10:40] = 1.0
    candidates.append(plateau)
    # Random
    for _ in range(15):
        candidates.append(np.abs(rng.standard_normal(N)) + 0.05)

    # Phase 1: Nelder-Mead from each candidate
    for h0 in candidates:
        if time.time() > deadline - 15:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize(params0, 3000, 1e-7, 1e-9)

    # Phase 2: Polish best with L-BFGS-B
    if time.time() < deadline - 10:
        try_bfgs(np.log(best_h), 500)

    # Phase 3: More random restarts if time permits
    for _ in range(25):
        if time.time() > deadline - 8:
            break
        h0 = np.abs(rng.standard_normal(N)) + 0.01
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize(params0, 1500, 1e-7, 1e-9)

    # Phase 4: Final polish
    if time.time() < deadline - 3:
        try_bfgs(np.log(best_h), 300)

    best_h = best_h / best_h.sum()
    return list(best_h)
# EVOLVE-BLOCK-END
