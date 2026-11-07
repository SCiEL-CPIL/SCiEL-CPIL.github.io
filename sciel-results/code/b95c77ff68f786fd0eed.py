import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct
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

def _h_from_dct_coeffs(coeffs):
    """Convert DCT coefficients to nonnegative heights."""
    # coeffs: first K DCT-II coefficients
    # We use idct to reconstruct, then take absolute value and ensure nonnegativity
    n = N
    # Pad coefficients to full length with zeros
    full_coeffs = np.zeros(n)
    k = min(len(coeffs), n)
    full_coeffs[:k] = coeffs[:k]
    # idct with norm='ortho'
    h = idct(full_coeffs, type=2, norm='ortho')
    h = np.abs(h)  # enforce nonnegativity
    h = np.maximum(h, 0.0)
    return h

def _neg_value_dct(params):
    """Negative objective using DCT parameterization."""
    h = _h_from_dct_coeffs(params)
    return -_compute_value(h)

def _h_from_gauss(params):
    """Single Gaussian: h[i] = a * exp(-b*(x-c)^2) + d"""
    a, b, c, d = params
    x = np.linspace(-0.25, 0.25, N)
    h = a * np.exp(-b * (x - c)**2) + d
    return np.maximum(h, 0.0)

def _neg_value_gauss(params):
    h = _h_from_gauss(params)
    return -_compute_value(h)

def _h_from_bimodal(params):
    """Two Gaussians: a1*exp(-b1*(x-c1)^2) + a2*exp(-b2*(x-c2)^2)"""
    a1, b1, c1, a2, b2, c2 = params
    x = np.linspace(-0.25, 0.25, N)
    h = a1 * np.exp(-b1 * (x - c1)**2) + a2 * np.exp(-b2 * (x - c2)**2)
    return np.maximum(h, 0.0)

def _neg_value_bimodal(params):
    h = _h_from_bimodal(params)
    return -_compute_value(h)

def _h_from_poly(params):
    """Polynomial: sum_{k=0}^{4} a_k * x^k"""
    x = np.linspace(-0.25, 0.25, N)
    h = params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4
    return np.maximum(h, 0.0)

def _neg_value_poly(params):
    h = _h_from_poly(params)
    return -_compute_value(h)

def _try_optimize(neg_func, x0, bounds, deadline, max_iter_nm=1500, max_iter_lbfgs=300):
    """Run Nelder-Mead then L-BFGS-B polish, return (best_x, best_val)."""
    t0 = time.time()
    
    # Nelder-Mead
    try:
        res_nm = minimize(neg_func, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
        best_x = res_nm.x
        best_f = res_nm.fun
    except Exception:
        return x0, -neg_func(x0)
    
    t1 = time.time()
    if t1 < deadline - 3:
        try:
            res_lbfgs = minimize(neg_func, best_x, method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': max_iter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_lbfgs.fun < best_f:
                best_x = res_lbfgs.x
                best_f = res_lbfgs.fun
        except Exception:
            pass
    
    return best_x, -best_f

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Strategy 1: DCT coefficient optimization (low-dimensional)
    # Use first 8 DCT coefficients
    K_dct = 8
    bounds_dct = [(-5.0, 5.0)] * K_dct
    starts_dct = [
        np.array([1.0] + [0.0]*(K_dct-1)),
        np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.8, 0.3, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.2, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.2, -0.1, 0.1, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.9, -0.2, 0.15, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([1.1, 0.1, 0.0, -0.1, 0.0, 0.05, 0.0, 0.0]),
    ]
    
    for x0 in starts_dct:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _try_optimize(_neg_value_dct, x0, bounds_dct, deadline)
            if val > best_val:
                best_val = val
                best_h = _h_from_dct_coeffs(x_opt)
        except Exception:
            pass
    
    # Strategy 2: Single Gaussian
    bounds_g = [(0.1, 10.0), (1.0, 100.0), (-0.25, 0.25), (-5.0, 5.0)]
    starts_g = [
        [1.0, 10.0, 0.0, 0.0],
        [2.0, 5.0, 0.0, 0.0],
        [1.0, 20.0, 0.1, 0.0],
        [1.0, 20.0, -0.1, 0.0],
        [3.0, 8.0, 0.0, -0.5],
        [1.0, 15.0, 0.05, 0.1],
        [1.0, 15.0, -0.05, 0.1],
        [2.0, 12.0, 0.0, 0.0],
    ]
    
    for x0 in starts_g:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _try_optimize(_neg_value_gauss, x0, bounds_g, deadline)
            if val > best_val:
                best_val = val
                best_h = _h_from_gauss(x_opt)
        except Exception:
            pass
    
    # Strategy 3: Bimodal (two Gaussians)
    bounds_b = [(0.1, 10.0), (1.0, 100.0), (-0.25, 0.25),
                (0.1, 10.0), (1.0, 100.0), (-0.25, 0.25)]
    starts_b = [
        [1.0, 10.0, -0.1, 1.0, 10.0, 0.1],
        [1.0, 15.0, -0.15, 1.0, 15.0, 0.15],
        [2.0, 8.0, -0.1, 2.0, 8.0, 0.1],
        [1.0, 20.0, -0.2, 1.0, 20.0, 0.2],
        [1.5, 12.0, -0.12, 1.5, 12.0, 0.12],
        [1.0, 10.0, -0.05, 1.0, 10.0, 0.05],
        [2.0, 15.0, -0.1, 0.5, 15.0, 0.1],
        [0.5, 15.0, -0.1, 2.0, 15.0, 0.1],
    ]
    
    for x0 in starts_b:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _try_optimize(_neg_value_bimodal, x0, bounds_b, deadline)
            if val > best_val:
                best_val = val
                best_h = _h_from_bimodal(x_opt)
        except Exception:
            pass
    
    # Strategy 4: Polynomial
    bounds_p = [(-5.0, 5.0)] * 5
    starts_p = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 5.0, 0.0, 0.0],
        [0.5, 0.0, 3.0, 0.0, 1.0],
        [1.0, 1.0, 2.0, 0.0, 0.0],
        [1.0, -1.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 4.0, 0.5, -2.0],
        [0.8, 0.0, 3.0, -0.5, 1.5],
    ]
    
    for x0 in starts_p:
        if time.time() > deadline:
            break
        try:
            x_opt, val = _try_optimize(_neg_value_poly, x0, bounds_p, deadline)
            if val > best_val:
                best_val = val
                best_h = _h_from_poly(x_opt)
        except Exception:
            pass
    
    # Final polish: use the best h found and do a direct coordinate descent
    # on a few random perturbations to see if we can improve
    if time.time() < deadline - 10:
        best_h = np.maximum(best_h, 0.0)
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random perturbation
            h_pert = best_h + np.random.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0.0)
            val_pert = _compute_value(h_pert)
            if val_pert > best_val:
                best_val = val_pert
                best_h = h_pert
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
