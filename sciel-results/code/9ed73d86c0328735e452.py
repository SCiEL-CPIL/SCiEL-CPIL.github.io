import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # Objective: Minimize the peak autoconvolution
    # Subject to: sum(h) = 1, h >= 0
    
    def peak_autocorr(h):
        h_conv = np.convolve(h, h, mode='full')
        return np.max(h_conv)
    
    # Initial guess: Uniform distribution
    h0 = np.ones(N) / N
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1},
    ]
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = minimize(
            peak_autocorr, 
            h0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res.success:
            h_opt = res.x
            # Ensure non-negativity and normalization
            h_opt = np.maximum(h_opt, 0)
            sum_h = np.sum(h_opt)
            if sum_h > 0:
                h_opt = h_opt / sum_h
            else:
                h_opt = np.ones(N) / N
            return list(h_opt)
    except Exception:
        pass
    
    # Fallback to Uniform
    return list(np.ones(N) / N)
