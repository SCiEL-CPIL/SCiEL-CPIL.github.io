import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
from scipy.optimize import minimize

N = 14

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area
    # Number of triangles is C(N, 3) = 364 for N=14
    # We can compute all areas efficiently.
    min_area = float('inf')
    
    # Get indices
    idx = np.arange(N)
    # This loop is fine for N=14
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0

    return min_area / hull_area

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We keep a safety margin.
    deadline = start_time + 145.0
    
    # Initial guess: Regular N-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    best_points = initial_points.copy()
    best_score = get_score(best_points)
    
    # Optimization
    # We want to maximize score, so minimize -score
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -get_score(pts)
        
    initial_flat = initial_points.flatten()
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, initial_flat, method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000})
        if res.fun < -best_score: # res.fun is -score, so -res.fun > best_score
            best_score = -res.fun
            best_points = res.x.reshape(N, 2)
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 5.0:
        return best_points

    # Try local search with random perturbations if time permits
    # This helps escape local minima or refine further
    current_points = best_points.copy()
    current_score = best_score
    step_size = 0.05
    
    # Vectorized score calculation for faster local search
    def fast_score(points):
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Compute all triangle areas vectorized
        # points shape (N, 2)
        # We need C(N,3) triangles
        # Generate indices
        i_idx, j_idx, k_idx = np.indices((N, N, N))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        p1 = points[i_flat]
        p2 = points[j_flat]
        p3 = points[k_flat]
        
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        if min_area < 1e-10:
            return 0.0
            
        return min_area / hull_area

    while time.time() < deadline - 2.0:
        improved = False
        # Try moving each point randomly
        for i in range(N):
            for _ in range(10): # 10 random tries per point
                if time.time() > deadline - 2.0:
                    break
                new_point = current_points[i] + np.random.normal(0, step_size, 2)
                trial_points = current_points.copy()
                trial_points[i] = new_point
                trial_score = fast_score(trial_points)
                if trial_score > current_score:
                    current_points = trial_points
                    current_score = trial_score
                    improved = True
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        if not improved:
            # Reduce step size or break
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    return best_points
