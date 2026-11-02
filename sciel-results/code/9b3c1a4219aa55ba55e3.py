import numpy as np
from scipy.optimize import minimize

N = 600

def _autoconvolve(h):
    """Compute the autoconvolution of a sequence."""
    return np.convolve(h, h, mode='full')

def _objective(params):
    """Objective function: normalized maximum of autoconvolution."""
    h = np.clip(params, 0, None)
    # Normalize to unit L2 norm for scale invariance
    norm = np.linalg.norm(h)
    if norm < 1e-12:
        return 1e6
    h_norm = h / norm
    ac = _autoconvolve(h_norm)
    return np.max(ac)

def construct():
    # Start with a random initialization
    np.random.seed(42)
    x0 = np.random.rand(N)
    
    # Use L-BFGS-B for bounded optimization
    bounds = [(0, 1)] * N
    
    # We want to minimize the peak of the autoconvolution
    result = minimize(
        _objective, 
        x0, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
    )
    
    h = np.clip(result.x, 0, None)
    
    # Ensure it's not all zeros
    if np.sum(h) == 0:
        h[N//2] = 1.0
    
    return list(h)
