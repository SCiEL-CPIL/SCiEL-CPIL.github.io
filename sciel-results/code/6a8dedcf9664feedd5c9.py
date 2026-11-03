"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_herm
import time

_START_TIME = time.time()
_DEADLINE = 175.0  # Safety margin

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_coeffs(c0, c4, c8):
    """Return [c0, c4, c8, c12] with g(0)=0 constraint."""
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = -665280.0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    return np.array([c0, c4, c8, c12])

def _compute_bound(c0, c4, c8):
    """Compute the upper bound value to minimize."""
    coeffs = _hermite_coeffs(c0, c4, c8)
    
    # Numerical integration on a fine grid
    x = np.linspace(-12, 12, 20000)
    dx = x[1] - x[0]
    
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    poly = (coeffs[0] * eval_herm(0, x) + 
            coeffs[1] * eval_herm(4, x) + 
            coeffs[2] * eval_herm(8, x) + 
            coeffs[3] * eval_herm(12, x))
    g = poly * np.exp(-x**2 / 2)
    
    abs_g = np.abs(g)
    norm1 = np.trapz(abs_g, x)
    norm2_sq = np.trapz(g**2, x)
    
    if norm1 < 1e-15 or norm2_sq < 1e-15:
        return 1e10
    
    # A(f) = ||f||_2^2 / ||f||_1^2, and A(Ff) = A(f) since F(g)=g
    A_f = norm2_sq / (norm1**2)
    bound = A_f * A_f  # A(f) * A(Ff) = A_f^2
    return bound

def construct():
    """Return three coefficients minimizing the bound."""
    if _time_remaining() < 5:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _compute_bound(*seed)
    best_x = seed.copy()
    
    # Local optimization from seed with multiple methods
    methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell']
    for method in methods:
        if _time_remaining() < 10:
            break
        try:
            opts = {'maxiter': 1000}
            if method == 'Nelder-Mead':
                opts.update({'xatol': 1e-10, 'fatol': 1e-12})
            elif method == 'L-BFGS-B':
                opts.update({'gtol': 1e-10})
            
            res = minimize(_compute_bound, best_x, method=method, options=opts)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Small random restarts
    rng = np.random.default_rng(123)
    for scale in [0.01, 0.05, 0.1]:
        for _ in range(5):
            if _time_remaining() < 5:
                break
            x0 = best_x + rng.normal(0, scale, size=3)
            try:
                res = minimize(_compute_bound, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END

