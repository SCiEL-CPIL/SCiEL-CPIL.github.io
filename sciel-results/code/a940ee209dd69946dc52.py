"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2.0 * x
    for k in range(2, n + 1):
        H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    return H_nm1

def _objective(coeffs):
    """Compute upper bound C4 on A(f)*A(Fhat(f)) for the Hermite-Gaussian test function."""
    c0, c4, c8 = coeffs
    if not np.all(np.isfinite(coeffs)):
        return 1e10
    
    # Use a fine grid
    x = np.linspace(-12, 12, 15001)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    # g(0) = 0: c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = poly * np.exp(-x**2 / 2.0)
    
    g2 = g**2
    
    norm2 = np.trapz(g2, x)
    if norm2 < 1e-25:
        return 1e10
    
    # x^2 moment (variance-like quantity)
    x2_moment = np.trapz(x**2 * g2, x) / norm2
    
    # For the Fourier transform: F{H_n(x) exp(-x^2/2)} = (-i)^n H_n(xi) exp(-xi^2/2)
    # For even n, (-i)^n is real: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1
    # So the Fourier transform has the same polynomial structure
    # Thus the xi^2 moment equals the x^2 moment
    
    # The uncertainty product bound:
    # C4 = (∫x²|g|² / ∫|g|²) * (∫ξ²|ĝ|² / ∫|ĝ|²)
    # = x2_moment * xi2_moment = x2_moment^2 (by symmetry)
    
    return x2_moment**2

def construct():
    deadline = time.time() + 170  # Safety margin before 180s limit
    
    # Seed value
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeffs = seed_coeffs.copy()
    best_val = _objective(seed_coeffs)
    
    # Nelder-Mead from seed
    try:
        res = minimize(_objective, seed_coeffs, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_coeffs
    
    # Random restarts
    rng = np.random.default_rng(12345)
    for i in range(30):
        if time.time() > deadline - 15:
            break
        start = rng.normal(0, 0.05, 3)
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    # Final polish with SLSQP
    try:
        res = minimize(_objective, best_coeffs, method='SLSQP',
                      options={'maxiter': 500, 'ftol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END

