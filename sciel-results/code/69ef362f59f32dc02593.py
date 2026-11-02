"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(-1, 2)
    return -_score_points(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(13)
    
    # Generate diverse initializations
    inits = []
    
    # Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular polygon with jitter
    for jitter in [0.01, 0.05, 0.1]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # Random
    for _ in range(20):
        inits.append(rng.rand(N, 2))
        
    for init_pts in inits:
        if time.time() > deadline - 5.0:
            break
            
        init_params = init_pts.flatten()
        
        # Try Nelder-Mead
        try:
            res = minimize(
                _neg_score,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
            
        # Try Powell
        if time.time() > deadline - 10.0:
            continue
            
        try:
            res_powell = minimize(
                _neg_score,
                init_params,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            candidate_pts = res_powell.x.reshape(-1, 2)
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
