"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/4)
    where c12 is chosen so that g(0) = 0.
    
    The bound is: C4 = (int x^4 |g|^2 dx) * (int xi^4 |G(xi)|^2 dxi) / ||g||^4
    where G is the Fourier transform of g (angular frequency convention).
    """
    # c12 from g(0) = 0:
    # H0(0)=1, H4(0)=3, H8(0)=15, H12(0)=10395
    c12 = -(c0 + 3*c4 + 15*c8) / 10395.0
    
    # Use a fine grid for accurate integration
    x = np.linspace(-25, 25, 100001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    x2 = x**2
    H0 = np.ones_like(x)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 24*x2**3 + 120*x2**2 - 120*x2 + 15
    H12 = x2**6 - 60*x2**5 + 900*x2**4 - 3600*x2**3 + 30240*x2**2 - 45360*x2 + 10395
    
    g = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x2/4)
    
    # ||g||^2
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-30:
        return np.inf
    
    # int x^4 |g|^2 dx
    fourth_moment_x = np.trapz(x**4 * g**2, x)
    
    # Fourier transform using FFT
    n = len(x)
    nfft = 1 << (n - 1).bit_length()
    if nfft < 2 * n:
        nfft = 1 << (2 * n - 1).bit_length()
    
    g_pad = np.zeros(nfft, dtype=complex)
    g_pad[:n] = g
    
    G = dx * np.fft.fft(g_pad)
    xi = 2*np.pi*np.fft.fftfreq(nfft, d=dx)
    
    # int xi^4 |G(xi)|^2 dxi
    fourth_moment_xi = np.trapz(xi**4 * np.abs(G)**2, xi)
    
    C4 = (fourth_moment_x * fourth_moment_xi) / (norm_sq**2)
    
    return C4


def _objective(coeffs):
    """Objective function: the upper bound C4."""
    c0, c4, c8 = coeffs
    try:
        val = _compute_upper_bound_C4(c0, c4, c8)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """Return the three coefficients [c0, c4, c8] that minimize C4."""
    start_time = time.time()
    deadline = start_time + 175.0
    
    # Start from the seed solution
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_upper_bound_C4(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Local refinement using L-BFGS-B from seed
    remaining = deadline - time.time()
    if remaining > 10:
        res = minimize(_objective, best_coeffs, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    
    # Try several other starting points
    start_points = [
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.0, 0.0005]),
        np.array([-0.5, 0.1, 0.0001]),
        np.array([-2.0, -0.1, 0.001]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00025]),
        np.array([-1.3, 0.02, 0.0006]),
        np.array([-1.13, 0.04, 0.000308]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        res = minimize(_objective, sp, method='L-BFGS-B',
                      options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    
    # Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        res = minimize(_objective, best_coeffs, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
