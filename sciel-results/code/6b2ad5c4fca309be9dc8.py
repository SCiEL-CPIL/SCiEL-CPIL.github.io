"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: 180s limit, use 170s for safety
_DEADLINE = time.time() + 170

def _hermite(n, x):
    """Compute physicist's Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1

def _build_g(a0, a4, a8, x):
    """Build g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2) with g(0)=0."""
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    # g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 3024.0
    H12_0 = 10321920.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    return poly * np.exp(-x**2 / 2.0)

def _compute_bound(a0, a4, a8):
    """Compute the upper bound C4 for given coefficients."""
    # Use a sufficiently large domain
    x = np.linspace(-15, 15, 10000)
    g = _build_g(a0, a4, a8, x)
    
    # A(g) is likely the L1 norm or integral of |g|
    # Let's try L1 norm
    # Use trapezoid for NumPy >= 2.0 compatibility (trapz removed)
    if hasattr(np, 'trapezoid'):
        A_g = np.trapezoid(np.abs(g), x)
    else:
        A_g = np.trapz(np.abs(g), x)
    
    # Fourier transform of g
    # For numerical FT, use FFT on a uniform grid
    # But we need to be careful with normalization
    # Let's use scipy.fft
    from scipy.fft import fft, fftshift, fftfreq
    
    # Use a larger grid for better FT resolution
    N = 65536
    x_ft = np.linspace(-20, 20, N)
    g_ft = _build_g(a0, a4, a8, x_ft)
    
    dx = x_ft[1] - x_ft[0]
    freqs = fftfreq(N, d=dx)
    
    G = fft(g_ft) * dx
    G = fftshift(G)
    freqs = fftshift(freqs)
    
    # A(Fourier(g))
    if hasattr(np, 'trapezoid'):
        A_G = np.trapezoid(np.abs(G), freqs)
    else:
        A_G = np.trapz(np.abs(G), freqs)
    
    # The upper bound C4 = A(g) * A(Fourier(g))
    # But we need to be careful about normalization
    # Let's try the product
    bound = A_g * A_G
    
    # Check if coefficients are reasonable
    if not np.isfinite(bound) or bound < 0:
        return 1e10
    
    return bound

def construct():
    # Initial guess from prior art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_result = x0
    best_bound = _compute_bound(*x0)
    
    # Try a few local optimizations from different starting points
    starts = [
        x0,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.1, 0.045, 0.00035]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(
                _compute_bound,
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.success and res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Also try a grid search around the best result
    if time.time() < _DEADLINE - 15:
        a0, a4, a8 = best_result
        for da0 in [-0.1, -0.05, 0, 0.05, 0.1]:
            for da4 in [-0.01, 0, 0.01]:
                for da8 in [-0.00005, 0, 0.00005]:
                    if time.time() > _DEADLINE - 5:
                        break
                    trial = np.array([a0 + da0, a4 + da4, a8 + da8])
                    b = _compute_bound(*trial)
                    if b < best_bound:
                        best_bound = b
                        best_result = trial
    
    return best_result
# EVOLVE-BLOCK-END
