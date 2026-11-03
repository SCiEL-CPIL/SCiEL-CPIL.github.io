"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound(c0, c4, c8, x):
    """Compute the upper bound given coefficients and grid."""
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    # g(0) = 0 constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * 1 + c4 * 12 + c8 * 1680) / 40320
    
    gauss = np.exp(-x**2 / 2)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # Compute norms
    dx = x[1] - x[0]
    norm2 = np.sum(g**2) * dx
    x2_norm = np.sum(x**2 * g**2) * dx
    
    # Fourier transform using FFT
    N = len(x)
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
    
    # Frequency grid
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    freq2_norm = np.sum(freqs**2 * np.abs(G)**2) * dx
    
    if norm2 <= 0:
        return np.inf
    
    delta_x = np.sqrt(x2_norm / norm2)
    delta_xi = np.sqrt(freq2_norm / norm2)
    
    return delta_x * delta_xi


def construct():
    start_time = time.time()
    deadline = 180  # seconds
    
    # Setup grid
    x = np.linspace(-20, 20, 4096)
    
    # Initial guess from prior work
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    best_bound = _compute_bound(c0_init, c4_init, c8_init, x)
    
    # Objective function
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return best_bound
        c0, c4, c8 = coeffs
        return _compute_bound(c0, c4, c8, x)
    
    # Try multiple optimization methods and starting points
    start_points = [
        [c0_init, c4_init, c8_init],
        [c0_init * 1.1, c4_init * 0.9, c8_init * 1.2],
        [c0_init * 0.9, c4_init * 1.1, c8_init * 0.8],
        [-1.0, 0.05, 0.0003],
        [-1.2, 0.04, 0.0002],
    ]
    
    for sp in start_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
