"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate min triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _score(x0)
    best_points = x0.copy()
    
    # Try a few different initial configurations if time permits
    configs = [
        x0,
        x0 + 0.1 * np.random.randn(N, 2),
        # Two rings: 7 outer, 7 inner
        np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                             np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))]),
            np.column_stack([0.5*np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7), 
                             0.5*np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)])
        ])
    ]
    
    for x_init in configs:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(_neg_score, x_init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if time.time() > deadline:
                break
                
            if res.fun < -best_score:
                p_opt = res.x.reshape(-1, 2)
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = p_opt
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
