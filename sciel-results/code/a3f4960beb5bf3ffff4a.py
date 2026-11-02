"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
from scipy.optimize import minimize
import time


def _hermite_eval(n, x):
    """Evaluate physicist's Hermite polynomial H_n at x."""
    H = hermite(n)
    return H(x)


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound C4 for given coefficients.
    a12 is determined by g(0) = 0.
    """
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 13440.0
    H12_0 = 10321920.0  # H_12(0)
    
    # a12 from g(0) = 0: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = [0, 4, 8, 12]
    
    # Evaluate g(x) on a grid
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    
    g = np.zeros_like(x)
    for c, n in zip(coeffs, degrees):
        g += c * _hermite_eval(n, x) * np.exp(-x**2 / 2.0)
    
    # Fourier transform of g: since ĝ of H_n(x)e^{-x^2/2} is (i)^n H_n(x)e^{-x^2/2}
    g_hat = np.zeros_like(x)
    for c, n in zip(coeffs, degrees):
        g_hat += c * (1j)**n * _hermite_eval(n, x) * np.exp(-x**2 / 2.0)
    
    # Compute norms
    norm_g_sq = np.sum(np.abs(g)**2) * dx
    norm_g4 = np.sum(np.abs(g)**4) * dx
    norm_ghat_sq = np.sum(np.abs(g_hat)**2) * dx
    norm_ghat4 = np.sum(np.abs(g_hat)**4) * dx
    
    if norm_g_sq < 1e-15 or norm_ghat_sq < 1e-15:
        return 1e10
    
    # Bound C4: use ratio involving L4 and L2 norms
    # A(f) might be ||f||_4 / ||f||_2 or similar
    A_g = (norm_g4 / norm_g_sq**2) ** 0.25 if norm_g_sq > 0 else 1e10
    A_ghat = (norm_ghat4 / norm_ghat_sq**2) ** 0.25 if norm_ghat_sq > 0 else 1e10
    
    bound = A_g * A_ghat
    return bound


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns array of three coefficients.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s limit
    
    # Initial guess from prior work
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_bound = _compute_bound(*x0)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    n_starts = 50
    
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        if i == 0:
            x_init = x0.copy()
        else:
            # Perturb around initial guess or random starts
            if i < 20:
                x_init = x0 + rng.normal(0, 0.1, size=3)
            else:
                x_init = rng.uniform(-2, 2, size=3)
        
        try:
            result = minimize(
                lambda x: _compute_bound(*x),
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.success or result.fun < best_bound:
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
                    
        except Exception:
            continue
    
    # Final local refinement
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                lambda x: _compute_bound(*x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
