import numpy as np
from scipy.optimize import minimize
import time

N = 50

def objective(x):
    c = x.reshape(-1, 2)
    # Check boundaries
    r_bound = np.min(0.5 - np.abs(c))
    if r_bound < 0:
        return 1.0
    # Vectorized distance
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_dist = np.min(dists) / 2
    r = min(r_bound, r_dist)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initialization
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Candidate positions for the 50th point
    candidates = [
        np.array([1/14, 1/14]),
        np.array([-1/14, 1/14]),
        np.array([1/14, -1/14]),
        np.array([-1/14, -1/14]),
        np.array([0, 1/14]),
        np.array([0, -1/14]),
        np.array([1/14, 0]),
        np.array([-1/14, 0])
    ]
    
    best_x = None
    best_score = -1.0
    
    for p50 in candidates:
        centers = np.vstack([grid, p50])
        x0 = centers.flatten()
        
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_x = res.x
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to initial grid + first candidate
        centers = np.vstack([grid, candidates[0]])
        best_x = centers.flatten()
        
    return best_x.reshape(-1, 2)
