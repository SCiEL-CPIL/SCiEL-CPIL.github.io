import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute probabilists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        h0 = np.ones_like(x)
        h1 = x.copy()
        for k in range(2, n + 1):
            h2 = x * h1 - (k - 1) * h0
            h0, h1 = h1, h2
        return h1

def _trapz(y, x):
    """Compatible trapezoidal integration."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)

def _compute_bound(a0, a4, a8):
    """Compute the upper bound A(g)*A(ĝ) for the Hermite-Gaussian combination."""
    # H12 coefficient to ensure g(0) = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 10395.0
    
    # Grid for numerical integration
    # Reduced to 10000 points for speed, still high precision for this smooth function
    x = np.linspace(-10, 10, 10000)
    
    # Hermite polynomials
    h0 = _hermite(0, x)
    h4 = _hermite(4, x)
    h8 = _hermite(8, x)
    h12 = _hermite(12, x)
    
    # Gaussian weight
    gauss = np.exp(-x**2 / 2.0)
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    g = (a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12) * gauss
    
    # ||g||_1
    norm_g = _trapz(np.abs(g), x)
    
    # For even-degree Hermite polynomials, the Fourier transform (unitary)
    # satisfies \hat{g}(\xi) = g(\xi) because (-i)^k = 1 for k in {0, 4, 8, 12}.
    # Thus ||\hat{g}||_1 = ||g||_1.
    norm_gf = norm_g
    
    # The bound is ||g||_1 * ||ĝ||_1
    bound = norm_g * norm_gf
    
    return bound

def construct():
    # Starting point from the reference
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Initial bound
    best_x = x0.copy()
    best_bound = _compute_bound(*x0)
    
    start_time = time.time()
    deadline = start_time + 150.0  # Leave safety margin before 180s limit
    
    # Try multiple optimization methods and starting points
    starting_points = [
        x0,
        np.array([-1.0, 0.01, 0.0001]),
        np.array([-1.2, 0.02, 0.0002]),
        np.array([-0.9, 0.005, 0.00005]),
        np.array([-1.1, 0.015, 0.00015]),
        np.array([-1.05, 0.012, 0.00012]),
    ]
    
    for sp in starting_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _compute_bound,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.success or result.fun < best_bound:
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Try L-BFGS-B with gradient (numerical)
    if time.time() < deadline - 20:
        try:
            result = minimize(
                _compute_bound,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _compute_bound,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
