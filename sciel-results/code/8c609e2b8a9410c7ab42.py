"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    Uses vectorized computation for speed.
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if j + 1 < n:
                pk = points[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _objective_full(x):
    """
    Objective function for full coordinate optimization.
    x: (2*N,) array.
    """
    points = x.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_points = None
    
    # Regular polygon initialization
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Try multiple diverse starting points with Powell optimization
    # Powell is more robust for this non-smooth objective
    
    starting_points = []
    
    # 1. Regular polygon
    starting_points.append(init_points.copy())
    
    # 2. Slightly perturbed regular polygon
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = init_points + rng.normal(0, 0.1, init_points.shape)
        starting_points.append(perturbed)
    
    # 3. Two concentric rings (7 points each)
    angles_outer = 2 * np.pi * np.arange(7) / 7
    angles_inner = 2 * np.pi * np.arange(7) / 7 + np.pi / 7
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    starting_points.append(np.vstack([outer, inner]))
    
    # 4. Another ring configuration
    outer2 = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner2 = 0.6 * np.column_stack([np.cos(angles_outer + np.pi/7), np.sin(angles_outer + np.pi/7)])
    starting_points.append(np.vstack([outer2, inner2]))
    
    # 5. Random points in unit disk
    for _ in range(3):
        r = rng.uniform(0.3, 1.0, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        random_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        starting_points.append(random_pts)
    
    # Optimize each starting point with Powell
    for idx, start_pts in enumerate(starting_points):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(_objective_full, start_pts.flatten(), method='Powell',
                          options={'maxiter': 2000, 'maxfev': 10000})
            score = -res.fun
            points = res.x.reshape(N, 2)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    # If we have a good solution, try to refine it further with Nelder-Mead
    if best_points is not None and time.time() < deadline - 10:
        try:
            res_refine = minimize(_objective_full, best_points.flatten(), 
                                  method='Nelder-Mead',
                                  options={'maxiter': 3000, 'maxfev': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            score_refine = -res_refine.fun
            points_refine = res_refine.x.reshape(N, 2)
            
            if score_refine > best_score:
                best_score = score_refine
                best_points = points_refine.copy()
        except Exception:
            pass
    
    # Final check: if still no good solution, try one more time with a different approach
    if best_score < 0.01 and time.time() < deadline - 5:
        # Try radial optimization with fixed angles
        def _objective_radial(r):
            angles = 2 * np.pi * np.arange(N) / N
            points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            return -_score(points)
        
        init_r = np.ones(N)
        try:
            res_r = minimize(_objective_radial, init_r, method='Nelder-Mead',
                           options={'maxiter': 5000, 'maxfev': 5000})
            score_r = -res_r.fun
            points_r = np.column_stack([res_r.x * np.cos(angles), res_r.x * np.sin(angles)])
            
            if score_r > best_score:
                best_score = score_r
                best_points = points_r
        except Exception:
            pass
    
    if best_points is None:
        return init_points
        
    return best_points
# EVOLVE-BLOCK-END
