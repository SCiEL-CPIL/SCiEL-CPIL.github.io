"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

def _hermite_physicists_values(n, x):
    """Compute physicist's Hermite polynomial H_n(x) at points x."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return 2.0 * x
    elif n == 2:
        return 4.0 * x**2 - 2.0
    elif n == 3:
        return 8.0 * x**3 - 12.0 * x
    elif n == 4:
        return 16.0 * x**4 - 48.0 * x**2 + 12.0
    elif n == 5:
        return 32.0 * x**5 - 160.0 * x**3 + 120.0 * x
    elif n == 6:
        return 64.0 * x**6 - 480.0 * x**4 + 720.0 * x**2 - 120.0
    elif n == 7:
        return 128.0 * x**7 - 1344.0 * x**5 + 3360.0 * x**3 - 1680.0 * x
    elif n == 8:
        return 256.0 * x**8 - 4480.0 * x**6 + 26880.0 * x**4 - 53760.0 * x**2 + 13440.0
    elif n == 9:
        return 512.0 * x**9 - 12288.0 * x**7 + 107520.0 * x**5 - 302400.0 * x**3 + 151200.0 * x
    elif n == 10:
        return 1024.0 * x**10 - 38400.0 * x**8 + 537600.0 * x**6 - 2419200.0 * x**4 + 3024000.0 * x**2 - 113400.0
    elif n == 11:
        return 2048.0 * x**11 - 100352.0 * x**9 + 1848320.0 * x**7 - 10886400.0 * x**5 + 24948000.0 * x**3 - 13932000.0 * x
    elif n == 12:
        return 4096.0 * x**12 - 236544.0 * x**10 + 5644800.0 * x**8 - 54432000.0 * x**6 + 250800000.0 * x**4 - 469560000.0 * x**2 + 24192000.0
    else:
        raise ValueError(f"Unsupported Hermite degree {n}")

def _compute_value(c0, c4, c8):
    """Compute the upper bound value to minimize."""
    # Derive c12 from constraint P(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 13440.0
    H12_0 = 24192000.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Coefficients for the polynomial P(x)
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Let's collect powers of x
    
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4480x^6 + 26880x^4 - 53760x^2 + 13440
    # H12 = 4096x^12 - 236544x^10 + 5644800x^8 - 54432000x^6 + 250800000x^4 - 469560000x^2 + 24192000
    
    # Collect coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    a0 = c0 * 1.0 + c4 * 12.0 + c8 * 13440.0 + c12 * 24192000.0  # This should be 0
    a2 = c4 * (-48.0) + c8 * (-53760.0) + c12 * (-469560000.0)
    a4 = c4 * 16.0 + c8 * 26880.0 + c12 * 250800000.0
    a6 = c8 * (-4480.0) + c12 * (-54432000.0)
    a8 = c8 * 256.0 + c12 * 5644800.0
    a10 = c12 * (-236544.0)
    a12 = c12 * 4096.0
    
    # Numerical integration for A(f) = integral |g(x)| dx
    # and for the Fourier transform
    
    # Use a reasonable grid
    x = np.linspace(-12, 12, 4096)
    dx = x[1] - x[0]
    
    exp_term = np.exp(-x**2 / 2)
    
    # Compute P(x)
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    P = a0 + a2*x2 + a4*x4 + a6*x6 + a8*x8 + a10*x10 + a12*x12
    g = P * exp_term
    
    A_f = np.trapz(np.abs(g), x)
    
    # Fourier transform using FFT
    N = len(x)
    G = np.fft.fft(g) * dx
    G = np.fft.fftshift(G)
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    A_F = np.trapz(np.abs(G), k)
    
    return A_f * A_F

def construct():
    """Find optimal coefficients to minimize the upper bound."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin
    
    # Initial guess
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_value(*best_c)
    
    # Global optimization with differential evolution
    bounds = [(-3, 3), (-3, 3), (-3, 3)]
    
    try:
        result = differential_evolution(
            lambda c: _compute_value(*c),
            bounds=bounds,
            seed=42,
            maxiter=30,
            tol=1e-5,
            polish=False,
            init='sobol'
        )
        
        if result.fun < best_val:
            best_c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Local refinement
    if time.time() < deadline:
        try:
            result2 = minimize(
                lambda c: _compute_value(*c),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result2.fun < best_val:
                best_c = result2.x
                best_val = result2.fun
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
