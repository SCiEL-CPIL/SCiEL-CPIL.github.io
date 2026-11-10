import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    dim = n * 2
    bounds = [(-10, 10)] * dim
    
    def objective(params):
        pts = params.reshape(n, 2)
        # Centering helps symmetry but isn't strictly necessary for the ratio,
        # but it helps the optimizer explore relative positions.
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return -1e9
            
        # Maximize d_min / d_max
        # Minimize negative ratio
        return -(d_min / d_max)

    start_time = time.time()
    deadline = start_time + 590
    
    # Use DE
    # popsize=20, maxiter=100
    # We can increase these if time permits.
    
    best_result = None
    best_score = -1e9
    
    # We might run multiple instances or just one long one.
    # Let's try one robust run.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            seed=42
        )
        
        if result.success or result.fun < best_score:
            best_score = -result.fun # Convert back to ratio
            best_result = result.x
            
        # Check time
        if time.time() > deadline - 10:
             return best_result.reshape(n, 2) if best_result is not None else np.random.randn(n, 2)
             
        # If we have time, do a local refinement with Powell or Nelder-Mead
        # Powell is good for this.
        from scipy.optimize import minimize
        
        local_res = minimize(
            objective, 
            best_result, 
            method='Powell', 
            options={'maxiter': 1000}
        )
        
        if local_res.fun < best_score:
            best_score = -local_res.fun
            best_result = local_res.x
            
    except Exception:
        pass
    
    if best_result is None:
        # Fallback
        return np.random.randn(n, 2)
        
    return best_result.reshape(n, 2)
