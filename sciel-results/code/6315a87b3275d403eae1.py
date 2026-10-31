"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = np.convolve(h, h, mode='full')
    return np.max(np.abs(ac))

def _squared_integral(h):
    return np.sum(h) ** 2

def _objective(x, h0):
    """Objective: max|autoconv| / (sum h)^2, with penalty if sum is near zero."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    ac = np.convolve(h, h, mode='full')
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    safety_margin = 10.0
    
    best_h = np.ones(N)
    best_val = _autoconv_max_abs(best_h) / _squared_integral(best_h)
    
    # Try some analytical patterns first
    patterns = []
    # Ramp up then down (triangular)
    tri = np.concatenate([np.linspace(1, 0, N//2), np.linspace(0, 1, N - N//2)])
    patterns.append(tri)
    # Cosine taper
    x = np.arange(N)
    cos_taper = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    patterns.append(cos_taper)
    # Half ones, half negative
    half_neg = np.ones(N)
    half_neg[N//2:] = -1
    patterns.append(half_neg)
    # Quarter positive, quarter negative, quarter positive, quarter negative
    q = np.ones(N)
    q[N//4:2*N//4] = -1
    q[3*N//4:] = -1
    patterns.append(q)
    
    for p in patterns:
        s = np.sum(p)
        if abs(s) < 1e-6:
            continue
        val = _autoconv_max_abs(p) / _squared_integral(p)
        if val < best_val:
            best_val = val
            best_h = p.copy()
    
    # Try gradient-based optimization from best starting point
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 30:
        try:
            def obj(x):
                return _objective(x, best_h)
            
            res = minimize(obj, best_h, method='L-BFGS-B', 
                          options={'maxiter': 50, 'maxfun': 200, 'maxls': 20},
                          timeout=deadline - time.time() - safety_margin)
            if res.success or res.fun < best_val:
                h_opt = res.x
                s = np.sum(h_opt)
                if abs(s) > 1e-6:
                    val = _autoconv_max_abs(h_opt) / _squared_integral(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt
        except Exception:
            pass
    
    # Normalize: the actual value depends on scaling, but ratio is scale-invariant
    # Return best_h as the step heights
    return best_h
# EVOLVE-BLOCK-END
