import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
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

def _compute_bound(a0, a4, a8):
    """
    Compute the C4 bound: ||g||_4^2 / ||g||_2
    for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0.
    """
    # Hermite polynomial values at x=0 (physicists' convention)
    # H_{2k}(0) = (-1)^k * (2k)! / k!
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 + H4_0 * a4 + H8_0 * a8) / H12_0
    
    # Evaluate on a fine grid
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2.0)
    
    # L2 norm
    l2_sq = np.sum(g**2) * dx
    if l2_sq < 1e-30:
        return 1e10
    l2 = np.sqrt(l2_sq)
    
    # L4 norm: ||g||_4 = (integral |g|^4 dx)^{1/4}
    l4_integral = np.sum(g**4) * dx
    if l4_integral < 1e-30:
        return 1e10
    l4 = l4_integral ** 0.25
    l4_sq = l4_integral ** 0.5  # ||g||_4^2
    
    # C4 bound = ||g||_4^2 / ||g||_2
    c4 = l4_sq / l2
    
    return c4

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 210s limit
    
    # Known state-of-the-art coefficients (Goncalves et al. 2017)
    baseline = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = baseline.copy()
    best_val = _compute_bound(*baseline)
    
    # Try optimization with multiple starting points
    rng = np.random.default_rng(12345)
    
    # Define starting points
    starts = [baseline.copy()]
    
    # Add perturbed starts
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for _ in range(3):
            perturbation = scale * rng.standard_normal(3)
            starts.append(baseline + perturbation)
    
    # Try different optimization methods
    for i, x0 in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            # Nelder-Mead for derivative-free optimization
            res = minimize(
                _compute_bound,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        try:
            res = minimize(
                _compute_bound,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
