"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

# EVOLVE-BLOCK-START
def compute_metric(x):
    """Compute max|autoconv| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    # Linear autoconvolution
    autoconv = np.convolve(x, x, mode='full')
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)

def construct():
    deadline = time.time() + 120  # Safety margin before 150s limit
    
    # Strategy 1: All ones (baseline)
    x1 = np.ones(N)
    m1 = compute_metric(x1)
    
    # Strategy 2: Smooth window on ±1 pattern
    # Use a raised cosine window
    t = np.linspace(0, 1, N)
    window = 0.5 * (1 + np.cos(2 * np.pi * t))
    x2 = np.sign(np.sin(2 * np.pi * t)) * window
    # Ensure non-zero sum
    if abs(np.sum(x2)) < 1e-10:
        x2[0] += 0.1
    m2 = compute_metric(x2)
    
    # Strategy 3: Random ±1 with small DC offset
    np.random.seed(42)
    x3 = np.random.choice([-1, 1], size=N).astype(float)
    # Add small DC offset to ensure non-zero sum
    x3 += 0.01
    m3 = compute_metric(x3)
    
    # Strategy 4: Smooth positive sequence (Gaussian-like)
    t = np.linspace(-3, 3, N)
    x4 = np.exp(-t**2)
    m4 = compute_metric(x4)
    
    # Strategy 5: Try to optimize a parameterized family
    # Use a chirp-like phase with amplitude window
    best_x = x1
    best_m = m1
    
    candidates = [(x1, m1), (x2, m2), (x3, m3), (x4, m4)]
    for x, m in candidates:
        if m < best_m and np.isfinite(m):
            best_m = m
            best_x = x
    
    # Try a simple local optimization on a smooth parameterization
    if time.time() < deadline - 30:
        # Parameterize as x[i] = a * cos(2*pi*f*i/N + phi) + b
        # Optimize a, f, phi, b
        def param_to_x(params):
            a, f, phi, b = params
            i = np.arange(N)
            return a * np.cos(2 * np.pi * f * i / N + phi) + b
        
        def objective(params):
            x = param_to_x(params)
            return compute_metric(x)
        
        # Try a few initial points
        best_params = None
        best_obj = best_m
        
        for f0 in [0.1, 0.2, 0.5, 1.0, 2.0]:
            for a0 in [0.5, 1.0, 2.0]:
                for b0 in [0.0, 0.1, 0.5]:
                    for phi0 in [0, np.pi/2, np.pi]:
                        x0 = [a0, f0, phi0, b0]
                        try:
                            res = minimize(objective, x0, method='Nelder-Mead',
                                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
                            if res.fun < best_obj and np.isfinite(res.fun):
                                best_obj = res.fun
                                best_params = res.x
                        except:
                            pass
                        
                        if time.time() > deadline - 30:
                            break
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        
        if best_params is not None:
            best_x = param_to_x(best_params)
            best_m = best_obj
    
    return best_x
# EVOLVE-BLOCK-END
