"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_herm
from scipy.fft import fft, fftfreq
import time

def _hermite_coeffs():
    """Return H_n(0) for n=0,4,8,12 using physicists' Hermite polynomials."""
    # H_n(0) = (-1)^(n/2) * n! / (n/2)! for even n
    def h0(n):
        if n % 2 != 0:
            return 0.0
        k = n // 2
        # Compute n! / k!
        result = 1.0
        for i in range(k + 1, n + 1):
            result *= i
        return ((-1) ** k) * result
    return h0(0), h0(4), h0(8), h0(12)

H0_0, H4_0, H8_0, H12_0 = _hermite_coeffs()

def _construct_g(x, a0, a4, a8):
    """Construct g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)"""
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    H0 = np.ones_like(x)
    H4 = eval_herm(4, x)
    H8 = eval_herm(8, x)
    H12 = eval_herm(12, x)
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    return poly * np.exp(-x**2 / 2)

def _objective(coeffs):
    """Compute the upper bound: ||g||_1 * ||g_hat||_1"""
    a0, a4, a8 = coeffs
    x = np.linspace(-20, 20, 20000)
    dx = x[1] - x[0]
    g = _construct_g(x, a0, a4, a8)
    norm_g = np.trapz(np.abs(g), x)
    
    # Compute Fourier transform numerically
    # Use FFT with proper scaling
    n = len(x)
    G = fft(g)
    freqs = fftfreq(n, d=dx)
    # |G| is the magnitude of the Fourier transform
    # The L1 norm of the Fourier transform: integrate |G(f)| df
    # But we need to be careful with scaling
    norm_g_hat = np.trapz(np.abs(G), freqs)
    
    return norm_g * norm_g_hat

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, keep safety margin
    
    # Seed value
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_value = _objective(seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    # Try multiple starting points and optimization methods
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    random_starts = [
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00025]),
    ]
    
    for method in methods:
        for start in random_starts:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(_objective, start, method=method, options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if time.time() > deadline - 30:
                    break
                if res.success and res.fun < best_value:
                    best_value = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
        if time.time() > deadline - 30:
            break
    
    return best_coeffs
# EVOLVE-BLOCK-END

