import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_value(h):
    """Compute the objective value for given heights."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    points = np.linspace(-0.5, 0.5, m + 2)
    w = points[1] - points[0]
    l2 = 0.0
    for i in range(m + 1):
        l2 += (w / 3.0) * (y[i]**2 + y[i] * y[i+1] + y[i+1]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _param_to_heights(params):
    """Convert parameters to 50 nonnegative heights."""
    # params: [a, b, c, d] -> h[i] = max(0, a * exp(-b*(x_i-c)^2) + d)
    a, b, c, d = params
    x = np.linspace(-0.25, 0.25, N)
    h = a * np.exp(-b * (x - c)**2) + d
    h = np.maximum(h, 0.0)
    return h

def _param_to_heights_poly(params):
    """Convert parameters to 50 nonnegative heights using polynomial basis."""
    # params: [a0, a1, a2, a3, a4] -> h[i] = max(0, a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
    x = np.linspace(-0.25, 0.25, N)
    h = params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4
    h = np.maximum(h, 0.0)
    return h

def _param_to_heights_spline(params):
    """Convert parameters to 50 nonnegative heights using cubic spline-like basis."""
    # params: [a, b, c, d, e, f] for a sum of two Gaussians
    a1, b1, c1, a2, b2, c2 = params
    x = np.linspace(-0.25, 0.25, N)
    h = a1 * np.exp(-b1 * (x - c1)**2) + a2 * np.exp(-b2 * (x - c2)**2)
    h = np.maximum(h, 0.0)
    return h

def _neg_value(params):
    """Negative objective for minimization."""
    h = _param_to_heights(params)
    return -_compute_value(h)

def _neg_value_poly(params):
    h = _param_to_heights_poly(params)
    return -_compute_value(h)

def _neg_value_spline(params):
    h = _param_to_heights_spline(params)
    return -_compute_value(h)

def _optimize_from_start(neg_func, x0, bounds, deadline):
    """Run Nelder-Mead then L-BFGS-B polish."""
    t0 = time.time()
    # Nelder-Mead
    res_nm = minimize(neg_func, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
    best_x = res_nm.x
    best_f = res_nm.fun
    
    t1 = time.time()
    if t1 < deadline - 5:
        # L-BFGS-B polish
        try:
            res_lbfgs = minimize(neg_func, best_x, method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_lbfgs.fun < best_f:
                best_x = res_lbfgs.x
                best_f = res_lbfgs.fun
        except Exception:
            pass
    
    return best_x, -best_f

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Strategy 1: Single Gaussian parametric form
    # params: [a, b, c, d] with a>0, b>0, -0.25<=c<=0.25, d>=-a
    bounds1 = [(0.1, 10.0), (1.0, 100.0), (-0.25, 0.25), (-5.0, 5.0)]
    starts1 = [
        [1.0, 10.0, 0.0, 0.0],
        [2.0, 5.0, 0.0, 0.0],
        [1.0, 20.0, 0.1, 0.0],
        [1.0, 20.0, -0.1, 0.0],
        [3.0, 8.0, 0.0, -0.5],
        [1.0, 15.0, 0.05, 0.1],
        [1.0, 15.0, -0.05, 0.1],
        [2.0, 12.0, 0.0, 0.0],
    ]
    
    for x0 in starts1:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _optimize_from_start(_neg_value, x0, bounds1, deadline)
            if val > best_val:
                best_val = val
                best_h = _param_to_heights(x_opt)
        except Exception:
            pass
    
    # Strategy 2: Polynomial parametric form
    bounds2 = [(-5.0, 5.0)] * 5
    starts2 = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 5.0, 0.0, 0.0],
        [0.5, 0.0, 3.0, 0.0, 1.0],
        [1.0, 1.0, 2.0, 0.0, 0.0],
        [1.0, -1.0, 2.0, 0.0, 0.0],
    ]
    
    for x0 in starts2:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _optimize_from_start(_neg_value_poly, x0, bounds2, deadline)
            if val > best_val:
                best_val = val
                best_h = _param_to_heights_poly(x_opt)
        except Exception:
            pass
    
    # Strategy 3: Two-Gaussian parametric form
    bounds3 = [(0.1, 10.0), (1.0, 100.0), (-0.25, 0.25),
               (0.1, 10.0), (1.0, 100.0), (-0.25, 0.25)]
    starts3 = [
        [1.0, 10.0, -0.1, 1.0, 10.0, 0.1],
        [1.0, 15.0, -0.15, 1.0, 15.0, 0.15],
        [2.0, 8.0, -0.1, 2.0, 8.0, 0.1],
        [1.0, 20.0, -0.2, 1.0, 20.0, 0.2],
        [1.5, 12.0, -0.12, 1.5, 12.0, 0.12],
    ]
    
    for x0 in starts3:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _optimize_from_start(_neg_value_spline, x0, bounds3, deadline)
            if val > best_val:
                best_val = val
                best_h = _param_to_heights_spline(x_opt)
        except Exception:
            pass
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
