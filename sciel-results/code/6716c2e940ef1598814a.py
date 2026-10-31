import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve
import time

N = 600
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

def _metric(h):
    """Compute the score: max(autoconv) / sum(h^2)."""
    h = np.asarray(h, dtype=float)
    if np.sum(h**2) < 1e-10:
        return 1e10
    conv = fftconvolve(h, h, mode='full')
    max_conv = np.max(conv)
    l2_sq = np.sum(h**2)
    return max_conv / l2_sq

def _gmm_heights(params, K=6):
    """Convert GMM parameters to step function heights."""
    heights = np.zeros(N)
    for k in range(K):
        c = params[3*k]
        w = params[3*k+1]
        a = params[3*k+2]
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w ** 2))
    return heights

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    K = 6  # Number of Gaussian basis functions
    
    # Bounds for DE
    bounds = []
    for _ in range(K):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.01, 0.5))     # width
        bounds.append((0.0, 5.0))      # amplitude
    
    def objective(params):
        h = _gmm_heights(params, K)
        return _metric(h)
    
    best_h = None
    best_score = 1.0 # Seed score
    
    # Try a few different K values or seeds if time permits
    # Start with K=6
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50, # Keep low to ensure time safety
            tol=1e-6,
            popsize=15,
            init='sobol',
            polish=False,
            updating='immediate',
            workers=1
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            best_h = _gmm_heights(best_params, K)
            best_score = result.fun
            
        # Polish the best result found so far
        if time.time() < deadline - 20 and best_h is not None:
            # Use a local optimizer on the GMM parameters? 
            # Or just refine the heights directly?
            # Let's try a simple local search on the heights using the gradient from the seed code?
            # No, the structural change is the GMM. Let's just run DE again with more iterations if time permits.
            
            # Actually, let's try to refine the best GMM found using L-BFGS-B
            from scipy.optimize import minimize
            
            # We need to provide gradients or use numerical.
            # DE already did a good job. Let's just ensure we return the best.
            
    except Exception as e:
        pass
    
    # If DE failed or didn't improve, try a simpler heuristic or return seed
    if best_h is None:
        best_h = np.ones(N)
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
