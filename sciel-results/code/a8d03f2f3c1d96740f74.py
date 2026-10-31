import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit out of 120s
    
    # 1. Generate initial configuration
    # Hexagonal lattice points
    # Center
    points = [(0,0)]
    # Ring 1
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    points.extend(ring1)
    # Ring 2
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), 
             (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)]
    points.extend(ring2)
    
    # We have 19 points. Remove 3 to get 16.
    # Remove (2,0), (0,2), (-2,2)
    remove_set = {(2,0), (0,2), (-2,2)}
    points = [p for p in points if p not in remove_set]
    
    # Ensure we have exactly 16
    assert len(points) == 16, f"Expected 16 points, got {len(points)}"
    
    # Convert to numpy array
    x0 = np.array(points, dtype=float)
    
    # Objective function
    def objective(x):
        # x is flattened array of shape (16, 2)
        pts = x.reshape(-1, 2)
        # Calculate pairwise distances
        # Use scipy.spatial.distance.pdist for efficiency?
        # Or manual calculation for N=16 is fast enough
        d = []
        for i in range(N):
            for j in range(i+1, N):
                d.append(np.linalg.norm(pts[i] - pts[j]))
        d = np.array(d)
        d_min = np.min(d)
        d_max = np.max(d)
        
        # Avoid division by zero
        if d_min < 1e-9:
            return 1e10
        return (d_max / d_min) ** 2
        
    def gradient(x):
        # Numerical gradient or analytical?
        # Powell doesn't need gradient
        return None

    # Optimization
    best_score = objective(x0)
    best_x = x0.copy()
    
    # Try Powell
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline:
        return best_x.reshape(-1, 2)
        
    # Try Nelder-Mead on the result
    try:
        res2 = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
        if res2.fun < best_score:
            best_score = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return best_x.reshape(-1, 2)
