import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def _objective_de(h):
    """
    Objective function for Differential Evolution.
    Returns the score. If h is invalid (all zeros), returns a large penalty.
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative, though bounds should handle this
    h = np.maximum(h, 0)
    s = _score(h)
    if np.isinf(s) or np.isnan(s):
        return 1e10
    return s

def construct():
    # Bounds for each step height
    # Since the score is scale invariant, we can restrict to [0, 1]
    # This keeps the numbers stable and helps DE.
    bounds = [(0, 1)] * N
    
    # Use Differential Evolution
    # popsize=10 is a good balance for 600 dimensions
    # maxiter=100 allows some convergence
    # tol=1e-5
    # polish=True will run L-BFGS-B on the best result found by DE
    # This is a hybrid, but the global search is DE.
    
    try:
        result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=100,
            popsize=10,
            tol=1e-5,
            polish=True,
            seed=42,
            init='latinhypercube'
        )
        
        h_opt = result.x
        # Ensure non-negative
        h_opt = np.maximum(h_opt, 0)
        
        # Calculate final score
        final_score = _score(h_opt)
        
        # Fallback if DE fails or returns bad result
        if np.isnan(final_score) or final_score > 1.0:
             # Try a simple cosine bell
             x = np.linspace(-0.25, 0.25, N)
             h_cos = np.cos(np.pi * x / 0.5) ** 2
             h_cos = np.maximum(h_cos, 0)
             # Normalize to [0,1] range for consistency, though scale doesn't matter
             h_cos = h_cos / np.max(h_cos) if np.max(h_cos) > 0 else h_cos
             return h_cos.tolist()
             
        return h_opt.tolist()

    except Exception:
        # Fallback to a known good shape if DE crashes
        x = np.linspace(-0.25, 0.25, N)
        h_cos = np.cos(np.pi * x / 0.5) ** 2
        h_cos = np.maximum(h_cos, 0)
        h_cos = h_cos / np.max(h_cos) if np.max(h_cos) > 0 else h_cos
        return h_cos.tolist()
