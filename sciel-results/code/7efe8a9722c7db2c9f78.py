"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute the nth physicist's Hermite polynomial at x."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return 2 * x
    else:
        Hn_2 = np.ones_like(x, dtype=float)
        Hn_1 = 2 * x
        for k in range(2, n + 1):
            Hn = 2 * x * Hn_1 - 2 * (k - 1) * Hn_2
            Hn_2 = Hn_1
            Hn_1 = Hn
        return Hn_1

def _compute_c12(c0, c4, c8):
    """Compute c12 so that g(0) = 0."""
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 443520.0
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty product for the given coefficients."""
    c12 = _compute_c12(c0, c4, c8)
    
    # Use a fine grid
    x = np.linspace(-15, 15, 100001)
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    g_poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = g_poly * np.exp(-x**2 / 2)
    
    norm2 = np.sum(g**2) * (x[1] - x[0])
    if norm2 <= 1e-15:
        return 1e10
    
    moment2 = np.sum(x**2 * g**2) * (x[1] - x[0])
    
    # Compute derivative for Fourier side
    H3 = _hermite(3, x)
    H7 = _hermite(7, x)
    H11 = _hermite(11, x)
    g_poly_prime = c4 * 8 * H3 + c8 * 16 * H7 + c12 * 24 * H11
    g_prime = g_poly_prime * np.exp(-x**2/2) - x * g_poly * np.exp(-x**2/2)
    
    deriv_norm2 = np.sum(g_prime**2) * (x[1] - x[0])
    
    if deriv_norm2 <= 1e-15:
        return 1e10
    
    A_f = np.sqrt(moment2 / norm2)
    A_F = np.sqrt(deriv_norm2 / norm2)
    
    return A_f * A_F

def construct():
    start_time = time.time()
    deadline = start_time + 175  # Safety margin
    
    best_x = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_bound(*best_x)
    
    np.random.seed(123)
    
    # Multi-start optimization
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Different initialization strategies
        if i == 0:
            x0 = best_x.copy()
        elif i == 1:
            x0 = np.array([-1.0, 0.1, 0.001])
        elif i == 2:
            x0 = np.array([0.0, 0.0, 0.0])
        else:
            x0 = np.random.randn(3) * 0.3
        
        try:
            result = minimize(_compute_bound, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Polish
    if time.time() < deadline - 5:
        try:
            result = minimize(_compute_bound, best_x, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
