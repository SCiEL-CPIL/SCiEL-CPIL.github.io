import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute probabilists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return x.astype(float).copy()
    else:
        h0 = np.ones_like(x, dtype=float)
        h1 = x.astype(float).copy()
        for k in range(2, n + 1):
            h2 = x * h1 - (k - 1) * h0
            h0, h1 = h1, h2
        return h1

def _compute_bound(a0, a4, a8):
    """Compute the upper bound ||g||_1 * ||ĝ||_1 for the Hermite-Gaussian combination."""
    # H12 coefficient to ensure g(0) = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 10395.0
    
    # Use a fine grid for accurate integration
    x = np.linspace(-15, 15, 50000)
    
    # Hermite polynomials
    h0 = _hermite(0, x)
    h4 = _hermite(4, x)
    h8 = _hermite(8, x)
    h12 = _hermite(12, x)
    
    # Gaussian weight exp(-x^2/2)
    gauss = np.exp(-0.5 * x**2)
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    g = (a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12) * gauss
    
    # ||g||_1 using trapezoidal rule
    norm_g = np.trapz(np.abs(g), x)
    
    # For the unitary Fourier transform, FT[H_n(x)*exp(-x^2/2)] = i^n * H_n(x) * exp(-x^2/2)
    # So |ĝ(x)| = |g(x)|, meaning ||ĝ||_1 = ||g||_1
    # Thus the product is ||g||_1^2
    norm_gf = norm_g
    
    bound = norm_g * norm_gf
    return bound

def construct():
    # Starting point from the reference
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Initial bound
    best_x = x0.copy()
    best_bound = _compute_bound(*x0)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 180s limit
    
    # Multiple starting points for robustness
    starting_points = [
        x0,
        x0 + np.array([0.1, -0.001, 0.0001]),
        x0 - np.array([0.1, 0.001, 0.0001]),
        x0 + np.array([-0.05, 0.002, -0.0002]),
        np.array([-1.15, 0.04, 0.0003]),
        np.array([-1.10, 0.035, 0.00025]),
    ]
    
    for sp in starting_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _compute_bound,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Local refinement with L-BFGS-B
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _compute_bound,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 15:
        try:
            result = minimize(
                _compute_bound,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-16, 'fatol': 1e-17, 'adaptive': True}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
