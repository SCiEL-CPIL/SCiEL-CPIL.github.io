"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    K = np.convolve(h, h)
    return abs(2.0 * N * np.max(K) / (s * s))

def _value_from_params(params):
    """
    Parameters: [a, b, c]
    h[i] = a + b*(-1)^i + c*cos(2*pi*i/N)
    This gives a flexible family.
    """
    a, b, c = params
    i = np.arange(N)
    h = a + b * ((-1) ** i) + c * np.cos(2.0 * np.pi * i / N)
    return _value(h)

def _construct_from_params(params):
    a, b, c = params
    i = np.arange(N)
    h = a + b * ((-1) ** i) + c * np.cos(2.0 * np.pi * i / N)
    return h

def construct():
    """
    Use scipy.optimize to minimize the value over a parameterized family.
    h[i] = a + b*(-1)^i + c*cos(2*pi*i/N)
    
    The alternating term b*(-1)^i makes the autoconvolution have negative
    entries at certain lags, potentially reducing max(K) below sum(h[i]^2).
    """
    # Start from the seed and try to improve
    best_h = np.ones(N)
    best_val = _value(best_h)  # should be 2.0
    
    # Try several starting points
    start_points = [
        [1.0, 0.0, 0.0],      # seed
        [1.0, 0.5, 0.0],      # alternating
        [1.0, 0.9, 0.0],      # strong alternating
        [1.0, 0.99, 0.0],     # near-alternating
        [1.0, 0.9, 0.1],      # with cosine
        [1.0, 0.8, -0.1],     # negative cosine
        [1.0, 0.95, 0.05],    # small cosine
        [0.5, 0.5, 0.0],      # a=b (sum=0, invalid but nearby)
        [1.2, 0.8, 0.0],      # a > b
        [1.0, 0.99, 0.01],    # fine tuning
    ]
    
    for sp in start_points:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(
                _value_from_params,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = _construct_from_params(res.x)
        except Exception:
            pass
    
    # Try a second round with more refined starting points around best
    if time.time() < _DEADLINE - 20:
        a, b, c = _construct_from_params(np.zeros(3))  # placeholder
        # Use the best found params
        # Re-optimize with L-BFGS-B for finer convergence
        for sp in start_points[:5]:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(
                    _value_from_params,
                    sp,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_h = _construct_from_params(res.x)
            except Exception:
                pass
    
    # Final check: make sure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
