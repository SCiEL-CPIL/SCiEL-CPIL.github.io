import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _min_triangle_area_normalized(pts):
    """
    Computes the minimum triangle area for points normalized to:
    1. Centroid at (0,0)
    2. Convex Hull Area = 1
    Returns the min area. If invalid, returns -1.
    """
    # 1. Center
    pts = pts - np.mean(pts, axis=0)
    
    # 2. Check for degeneracy
    if np.allclose(pts, 0):
        return -1.0
        
    # 3. Compute Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return -1.0
        
    if hull_area < 1e-12:
        return -1.0
        
    # 4. Scale to unit area
    scale = 1.0 / np.sqrt(hull_area)
    pts_scaled = pts * scale
    
    # 5. Compute Min Triangle Area
    min_area = np.inf
    # N=14 is small, loop is fine
    for i in range(N):
        pi = pts_scaled[i]
        for j in range(i + 1, N):
            pj = pts_scaled[j]
            v1 = pj - pi
            for k in range(j + 1, N):
                pk = pts_scaled[k]
                v2 = pk - pi
                # Cross product 2D
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    return min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialize with Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    # Circumradius for unit area regular N-gon
    # Area = 0.5 * N * R^2 * sin(2*pi/N) = 1
    R = np.sqrt(1.0 / (0.5 * N * np.sin(2.0 * np.pi / N)))
    
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    init_points = np.column_stack((x, y))
    
    x0 = init_points.flatten()
    
    # Objective function to minimize: - (Min Triangle Area)
    # We use the normalized version to remove scale/translation invariance
    def objective(x_flat):
        pts = x_flat.reshape(N, 2)
        min_area = _min_triangle_area_normalized(pts)
        if min_area < 0:
            return 1e6 # Penalty for invalid
        return -min_area
        
    # Get initial score
    current_best_score = -objective(x0)
    current_best_x = x0
    
    # 2. Optimize using Powell
    # Powell is derivative-free and handles non-smooth minima well.
    try:
        # maxiter set to a reasonable number. Powell usually converges fast.
        res = minimize(
            objective, 
            x0, 
            method='Powell', 
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        # Check if we are still within time
        if time.time() < deadline:
            if res.fun < objective(x0): # res.fun is -area, so smaller is better
                current_best_x = res.x
                current_best_score = -res.fun
            else:
                # If Powell didn't improve, keep init
                pass
        else:
            # If timeout, keep whatever we had
            pass
            
    except Exception:
        pass # Keep initial

    # 3. Optional: Second phase with Nelder-Mead if time allows
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                objective, 
                current_best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res2.fun < objective(current_best_x):
                current_best_x = res2.x
                current_best_score = -res2.fun
        except Exception:
            pass

    return current_best_x.reshape(N, 2)
