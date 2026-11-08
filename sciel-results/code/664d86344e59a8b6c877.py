import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the evaluation value for given heights."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    # Points are on linspace(-0.5, 0.5, m+2), spacing w = 1.0/(m+1)
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def dct_to_heights(coeffs, N):
    """Convert DCT coefficients to nonnegative heights."""
    # Use DCT-II basis
    n = N
    x = np.arange(n)
    basis = np.cos(np.pi * x[:, None] * (2 * np.arange(n) + 1) / (2 * n))
    h = basis @ coeffs
    # Enforce nonnegativity
    h = np.maximum(h, 0.0)
    return h

def negative_value_dct(coeffs):
    """Negative of the value for minimization."""
    h = dct_to_heights(coeffs, N)
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with smooth DCT coefficients
    # Start with a bell-shaped profile in DCT space
    n_coeffs = N
    # Low-frequency emphasis for smoothness
    init_coeffs = np.zeros(n_coeffs)
    # Put energy in low frequencies
    for k in range(min(10, n_coeffs)):
        init_coeffs[k] = 1.0 / (1.0 + 0.5 * k)
    
    # Normalize initial heights to be reasonable
    h_init = dct_to_heights(init_coeffs, N)
    if np.sum(h_init) > 0:
        init_coeffs = init_coeffs * (1.0 / np.mean(h_init[h_init > 0]) if np.any(h_init > 0) else 1.0)
    
    best_coeffs = init_coeffs.copy()
    best_value = compute_value(dct_to_heights(best_coeffs, N))
    
    # Gradient-based optimization in DCT space
    try:
        result = minimize(
            negative_value_dct,
            init_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < -best_value:
            if -result.fun > best_value:
                best_coeffs = result.x
                best_value = -result.fun
    except Exception:
        pass
    
    # Try multiple restarts with different initial conditions
    rng = np.random.RandomState(42)
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random smooth initialization
        coeffs = np.zeros(n_coeffs)
        # Low-frequency random components
        for k in range(min(8, n_coeffs)):
            coeffs[k] = rng.randn() * (1.0 / (1.0 + 0.3 * k))
        
        h = dct_to_heights(coeffs, N)
        if np.sum(h) == 0:
            continue
        
        # Normalize
        coeffs = coeffs * (1.0 / np.mean(h[h > 0]) if np.any(h > 0) else 1.0)
        
        try:
            result = minimize(
                negative_value_dct,
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            val = -result.fun
            if val > best_value:
                best_value = val
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                negative_value_dct,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            if -result.fun > best_value:
                best_value = -result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Convert to heights
    h_final = dct_to_heights(best_coeffs, N)
    h_final = np.maximum(h_final, 0.0)
    
    # Ensure finite
    h_final = np.where(np.isfinite(h_final), h_final, 0.0)
    
    return list(h_final)
