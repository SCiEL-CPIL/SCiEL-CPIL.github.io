"""Optimized program for uncertainty inequality bound minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

# Precompute Hermite polynomials as coefficient arrays for fast evaluation
def _hermite_coeffs(n):
    """Get coefficients of the nth physicists' Hermite polynomial."""
    H = hermite(n)
    return H.coef()

# Precompute for n = 0, 4, 8, 12
_H0_c = _hermite_coeffs(0)
_H4_c = _hermite_coeffs(4)
_H8_c = _hermite_coeffs(8)
_H12_c = _hermite_coeffs(12)

# H_n(0) values
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = 10321920.0

def _compute_bound_fast(c0, c4, c8):
    """Compute the upper bound C4 efficiently."""
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0
    
    # Combine coefficients: p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # All are even polynomials, so p(x) = q(x^2)
    # p(x) = (c0 + c4*H4 + c8*H8 + c12*H12)
    
    # For efficiency, evaluate on a grid
    x = np.linspace(-12, 12, 20000)
    x2 = x**2
    
    # Evaluate polynomials
    H0 = np.ones_like(x)
    H4 = np.polyval(_H4_c, x)
    H8 = np.polyval(_H8_c, x)
    H12 = np.polyval(_H12_c, x)
    
    gauss = np.exp(-x2 / 2.0)
    
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # For even n, (-i)^n = (-1)^(n/2): n=0->1, n=4->1, n=8->1, n=12->1
    # So g_hat = g
    sup_g = np.max(np.abs(g))
    
    norm_sq = np.trapz(g**2, x)
    
    if norm_sq < 1e-30:
        return 1e30
    
    C4 = sup_g * sup_g / norm_sq
    return C4

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin
    
    # Initial guess from prior art
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    best_C4 = _compute_bound_fast(c0_init, c4_init, c8_init)
    best_params = np.array([c0_init, c4_init, c8_init])
    
    def objective(params):
        if time.time() > deadline - 20:
            return 1e30
        return _compute_bound_fast(params[0], params[1], params[2])
    
    # Multiple restarts
    np.random.seed(42)
    initial_points = [
        np.array([c0_init, c4_init, c8_init]),
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.08, 0.001]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0004]),
    ]
    
    for init in initial_points:
        if time.time() > deadline - 25:
            break
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7})
            if result.fun < best_C4:
                best_C4 = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Refine
    if time.time() < deadline - 25:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 100, 'gtol': 1e-9})
            if result.fun < best_C4:
                best_C4 = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
