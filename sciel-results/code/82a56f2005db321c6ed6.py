import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        # Check for degenerate points
        if len(points) < 3:
            return 0.0
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Compute Min Triangle Area
        # Number of triangles C(13, 3) = 286
        # Vectorized calculation
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial guesses
    starting_points = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_13 = np.column_stack((np.cos(angles), np.sin(angles)))
    starting_points.append(init_13)
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    init_12c = np.vstack([pts_12, np.array([[0.0, 0.0]])])
    starting_points.append(init_12c)
    
    # 3. Two rings: 7 outer, 6 inner
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r_inner = 0.5 # Arbitrary guess, will be optimized
    pts_7 = np.column_stack((np.cos(angles_7), np.sin(angles_7)))
    pts_6 = np.column_stack((r_inner * np.cos(angles_6), r_inner * np.sin(angles_6)))
    init_2ring = np.vstack([pts_7, pts_6])
    starting_points.append(init_2ring)
    
    # 4. Noisy versions of Regular 13-gon
    for _ in range(3):
        noise = rng.normal(0, 0.1, (N, 2))
        starting_points.append(init_13 + noise)
        
    # 5. Noisy versions of 12+Center
    for _ in range(2):
        noise = rng.normal(0, 0.1, (N, 2))
        starting_points.append(init_12c + noise)

    # Optimization loop
    for i, start in enumerate(starting_points):
        if time.time() > deadline:
            break
            
        # Try Powell method
        try:
            res = minimize(
                neg_score, 
                start.flatten(), 
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            cur_score = -res.fun
            if cur_score > best_score:
                best_score = cur_score
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            pass
        
        if time.time() > deadline:
            break
            
        # Try Nelder-Mead if time permits
        try:
            res = minimize(
                neg_score, 
                start.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            cur_score = -res.fun
            if cur_score > best_score:
                best_score = cur_score
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            pass

    # If no optimization worked or failed, return the regular polygon
    if best_points is None or best_score <= 0:
        best_points = init_13
    else:
        # Verify the best point is valid
        final_score = score(best_points)
        if final_score <= 0:
             best_points = init_13

    return best_points
