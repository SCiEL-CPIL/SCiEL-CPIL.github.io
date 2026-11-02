import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, fftfreq
import time

def _hermite_polys(x):
    """Compute H0, H4, H8, H12 evaluated at x."""
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = (x**12 - 60*x**10 + 1260*x**8 - 11340*x**6 
           + 30240*x**4 - 25200*x**2 + 10395)
    return H0, H4, H8, H12

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty product A(f)*A(F(f))."""
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    
    x = np.linspace(-12, 12, 32768)  # Power of 2 for FFT efficiency
    H0, H4, H8, H12 = _hermite_polys(x)
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g = P * np.exp(-x**2 / 2)
    
    dx = x[1] - x[0]
    
    # L2 norm squared
    norm_g_sq = np.sum(g**2) * dx
    if norm_g_sq < 1e-30:
        return 1e10
    
    # Variance in x-space
    pdf_g = g**2 / norm_g_sq
    mean_g = np.sum(x * pdf_g) * dx
    var_g = np.sum((x - mean_g)**2 * pdf_g) * dx
    
    # Fourier transform
    G = fft(g) * dx
    freqs = fftfreq(len(x), d=dx)
    
    norm_G_sq = np.sum(np.abs(G)**2) * dx
    if norm_G_sq < 1e-30:
        return 1e10
    
    # Variance in frequency space
    pdf_G = np.abs(G)**2 / norm_G_sq
    mean_G = np.sum(freqs * pdf_G) * dx
    var_G = np.sum((freqs - mean_G)**2 * pdf_G) * dx
    
    # Ensure positive
    var_g = max(var_g, 0)
    var_G = max(var_G, 0)
    
    return np.sqrt(var_g) * np.sqrt(var_G)

def construct():
    start_time = time.time()
    deadline = start_time + 185  # Safety margin
    
    # Seed values
    seed_c0, seed_c4, seed_c8 = -113/100, 1/25, 1/3240
    seed_value = _compute_bound(seed_c0, seed_c4, seed_c8)
    
    best_value = seed_value
    best_coeffs = np.array([seed_c0, seed_c4, seed_c8])
    
    def objective(coeffs):
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    # Try multiple random starting points with local optimization
    rng = np.random.default_rng(42)
    
    # First, try a coarse grid
    for c0 in np.linspace(-3, 3, 15):
        for c4 in np.linspace(-1, 1, 10):
            for c8 in np.linspace(-0.5, 0.5, 8):
                if time.time() > deadline:
                    break
                val = objective([c0, c4, c8])
                if val < best_value:
                    best_value = val
                    best_coeffs = np.array([c0, c4, c8])
    
    # Local refinement from best and several other points
    if time.time() < deadline:
        try:
            starts = [best_coeffs.copy()]
            for _ in range(10):
                starts.append(rng.uniform(-3, 3, size=3))
            
            for start in starts:
                if time.time() > deadline:
                    break
                res = minimize(
                    objective,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_coeffs = res.x
        except Exception:
            pass
    
    return best_coeffs
