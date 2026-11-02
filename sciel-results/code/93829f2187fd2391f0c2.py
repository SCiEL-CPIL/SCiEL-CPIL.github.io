"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

_start_time = time.time()
_DEADLINE = 210.0  # wall-clock limit


def _hermite_coeffs():
    """Return coefficients for H0, H4, H8, H12 as polynomial coefficient arrays (highest degree first)."""
    # Physicists' Hermite polynomials
    H0 = np.array([1.0])
    H4 = np.array([16, 0, -48, 0, 12])
    H8 = np.array([256, 0, -3072, 0, 11520, 0, -13440, 0, 1680])
    H12 = np.array([4096, 0, -98304, 0, 1032192, 0, -5806080, 0, 14417280, 0, -15482880, 0, 4790016])
    return H0, H4, H8, H12


def _compute_c12(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 4790016.0
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0


def _objective(coeffs):
    """Compute the upper bound on C4 given coefficients [c0, c4, c8]."""
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    H0, H4, H8, H12 = _hermite_coeffs()
    
    # Build g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Pad all to same length
    max_deg = 12
    g_coeffs = np.zeros(max_deg + 1)
    
    for c, poly in [(c0, H0), (c4, H4), (c8, H8), (c12, H12)]:
        deg = len(poly) - 1
        # poly is highest-degree first, so poly[i] corresponds to x^(deg-i)
        for i, coeff in enumerate(poly):
            power = deg - i
            g_coeffs[max_deg - power] += c * coeff
    
    # g(x) * exp(-x^2/2) - we need to compute norms
    # For the uncertainty inequality, we typically compute:
    # A(f) = ||f||_1 / ||f||_2 or similar
    # Let's use numerical integration
    
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    # g(x)
    g_vals = np.zeros_like(x)
    for i, coeff in enumerate(g_coeffs):
        power = max_deg - i
        g_vals += coeff * x**power
    
    gaussian = np.exp(-x**2 / 2)
    f = g_vals * gaussian
    
    # Compute L1 and L2 norms of f
    norm1 = np.sum(np.abs(f)) * dx
    norm2_sq = np.sum(f**2) * dx
    norm2 = np.sqrt(norm2_sq)
    
    if norm2 < 1e-15:
        return 1e10
    
    # Fourier transform of f(x) = g(x)*exp(-x^2/2)
    # The Fourier transform of exp(-x^2/2) is sqrt(2*pi)*exp(-k^2/2)
    # We can compute numerically
    k = np.linspace(-10, 10, 10000)
    dk = k[1] - k[0]
    
    # F(k) = integral f(x) exp(-ikx) dx
    # Use FFT-based approach or direct integration
    # For efficiency, use the fact that FT of polynomial*gaussian can be computed analytically
    # but for now, use numerical integration
    
    # Actually, let's use a simpler approach: the Fourier transform of x^n * exp(-x^2/2)
    # can be computed. But for simplicity, let's use numerical FFT
    
    # Pad to power of 2 for FFT
    npts = len(x)
    f_pad = np.zeros(2 * npts, dtype=complex)
    f_pad[:npts] = f
    
    # FFT
    F = np.fft.fft(f_pad) * dx
    
    # The FFT gives us F at frequencies k_j = 2*pi*j/(npts*dx) for j=0..npts-1
    # and negative frequencies for j=npts..2*npts-1
    # We need to extract the real part since f is real
    
    # For the uncertainty product, we need ||F||_1 / ||F||_2 or similar
    # Let's compute the norms
    F_abs = np.abs(F)
    norm1_F = np.sum(F_abs) * dx / (2 * np.pi)  # normalization for Fourier transform
    norm2_F_sq = np.sum(F_abs**2) * dx / (2 * np.pi)
    norm2_F = np.sqrt(norm2_F_sq)
    
    if norm2_F < 1e-15:
        return 1e10
    
    # The upper bound on C4 is related to (||f||_1/||f||_2) * (||F||_1/||F||_2)
    # or some similar combination
    ratio_f = norm1 / norm2
    ratio_F = norm1_F / norm2_F
    
    return ratio_f * ratio_F


def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed
    
    if remaining < 30:
        # Not enough time for optimization, return prior state of the art
        return np.array([-113 / 100, 1.0 / 25, 1.0 / 3240])
    
    # Start from prior state of the art
    x0 = np.array([-113 / 100, 1.0 / 25, 1.0 / 3240])
    
    best_coeffs = x0.copy()
    best_obj = _objective(x0)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 20:
        return best_coeffs
    
    # Try L-BFGS-B with bounds
    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        result2 = minimize(
            _objective,
            best_coeffs,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 3000,
                'gtol': 1e-8
            }
        )
        
        if result2.fun < best_obj:
            best_obj = result2.fun
            best_coeffs = result2.x
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
