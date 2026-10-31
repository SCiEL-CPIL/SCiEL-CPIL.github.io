import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200  # 200s deadline, 10s margin before 210s kill

def _hermite(x):
    """Compute H0, H4, H8, H12 (physicists' Hermite polynomials)."""
    x = np.asarray(x, dtype=np.float64)
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    return H0, H4, H8, H12

def _compute_supnorm(a0, a4, a8, x_samples):
    """Compute sup norm of g(x) = P(x) exp(-x^2/2) on x_samples."""
    H0, H4, H8, H12 = _hermite(x_samples)
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = P * np.exp(-x_samples**2 / 2)
    return np.max(np.abs(g))

def _objective(params, x_samples):
    """Objective: sup norm of g."""
    a0, a4, a8 = params
    return _compute_supnorm(a0, a4, a8, x_samples)

def construct():
    # Fine grid for sup norm approximation
    x_samples = np.linspace(-10, 10, 40001)
    
    # Seed initial guess
    best_params = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_supnorm(*best_params, x_samples)
    
    # Try multiple starting points with Nelder-Mead
    starts = [
        best_params,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.2, 0.15, 0.0005]),
        np.array([-0.8, 0.05, 0.002]),
        np.array([-1.5, 0.2, 0.0001]),
        np.array([-0.5, 0.08, 0.0015]),
        np.array([-1.1, 0.12, 0.0008]),
        np.array([-1.3, 0.18, 0.0003]),
        np.array([-0.9, 0.06, 0.0018]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE:
            break
        try:
            res = minimize(_objective, start, args=(x_samples,), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array(best_params)
