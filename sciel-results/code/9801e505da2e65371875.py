import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def _objective(x):
    """
    Objective function for differential evolution.
    x: array of length N, values in [0, 1].
    We convert to non-negative f, normalize to L1=1, and compute negative score.
    """
    f = np.asarray(x, dtype=float)
    # Ensure non-negativity (bounds should handle this, but just in case)
    f = np.maximum(f, 0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0 # Maximize score, so return 0 for invalid
    
    # Normalize to L1 = 1
    f = f / l1
    
    # Calculate Score
    l2_sq = np.sum(f**2)
    conv = np.convolve(f, f, mode='full')
    amax = np.max(conv)
    
    if amax < 1e-12:
        return 0.0
        
    score = l2_sq / amax
    return -score

def construct():
    # Bounds for each dimension
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Initial guess: uniform
    x0 = np.ones(N) / N
    
    # Run Differential Evolution
    # popsize=15, maxiter=1000 is a good balance for 50 dims
    # tol=1e-8
    # seed=42 for reproducibility
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=2000, 
            popsize=20, 
            tol=1e-10,
            polish=True,
            init='sobol' # Sobol sampling for better initial distribution
        )
        f_opt = np.maximum(res.x, 0)
        if np.sum(f_opt) > 1e-12:
            f_opt = f_opt / np.sum(f_opt)
        else:
            f_opt = np.ones(N) / N
    except Exception:
        f_opt = np.ones(N) / N

    # Polish with L-BFGS-B on the normalized vector
    # We need to handle the normalization constraint.
    # Simple approach: just use the result from DE which includes polish.
    # If polish failed or wasn't great, do a quick coordinate descent or L-BFGS on log-space?
    # For now, trust DE + Polish.
    
    # Calculate final score to ensure it's valid
    l1 = np.sum(f_opt)
    if l1 < 1e-12:
        f_opt = np.ones(N) / N
    else:
        f_opt = f_opt / l1
        
    return list(f_opt)
