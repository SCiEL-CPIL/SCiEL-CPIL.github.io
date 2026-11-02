"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 600
DELTA = 1.0 / (2 * N)
SUM_H = 1.0 / DELTA

def _get_conv_max(h):
    """Calculate the maximum of the autoconvolution of the step function defined by h."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    start = N // 2
    end = start + N
    conv_same = conv[start:end]
    return np.max(conv_same) * DELTA

def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    Uses bisection search on T and a feasibility check via SLSQP minimization.
    """
    h0 = np.ones(N) * (SUM_H / N)
    bounds = [(0, None) for _ in range(N)]
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - SUM_H, 'jac': lambda h: np.ones(N)}]
    
    low = 0.0
    high = 10.0
    T_uniform = _get_conv_max(h0)
    if T_uniform < high:
        high = T_uniform
        
    best_h = h0
    best_T = T_uniform
    
    for _ in range(20):
        mid = (low + high) / 2.0
        
        def objective(h):
            conv_max = _get_conv_max(h)
            return max(0.0, conv_max - mid)
        
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            if res.success and res.fun <= 1e-6:
                best_h = res.x
                best_T = mid
                high = mid
            else:
                low = mid
        except Exception:
            low = mid

    return list(best_h)
# EVOLVE-BLOCK-END
