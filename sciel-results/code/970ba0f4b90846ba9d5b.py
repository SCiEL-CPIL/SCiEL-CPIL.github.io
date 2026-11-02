"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # points shape (n, 2)
    # We need area for all triples i, j, k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
    
    # To keep it simple and robust within time limits, we use a loop but optimized numpy ops
    # For N=14, there are 364 triangles. A loop is fine.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback if points are collinear or few
        return 1e-9

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def construct():
    """
    Multi-start local optimization using Nelder-Mead.
    Starts from several diverse configurations and refines the best ones.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random points in unit square
    for _ in range(10):
        initial_configs.append(np.random.rand(N, 2))
        
    # 2. Points on a circle (regular polygon)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. Grid-like points (distorted)
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    if len(grid_points) >= N:
        # Shuffle and take first N
        idx = np.random.choice(len(grid_points), N, replace=False)
        initial_configs.append(grid_points[idx])
        
    # 4. Two concentric circles
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    r1, r2 = 0.3, 0.7
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    initial_configs.append(np.vstack([pts1, pts2]))

    # Refine each configuration
    for i, init_points in enumerate(initial_configs):
        if time.time() - start_time > time_limit:
            break
            
        # Flatten points for Nelder-Mead
        x0 = init_points.flatten()
        
        # Objective function to minimize (negative score)
        def objective(x):
            points = x.reshape(N, 2)
            return -_score(points)
            
        try:
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            refined_points = result.x.reshape(N, 2)
            current_score = _score(refined_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = refined_points.copy()
                
        except Exception:
            pass

    # If we still have time, do a few more random restarts from the best found
    if best_points is not None and time.time() - start_time < time_limit:
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            # Perturb best points
            noise = np.random.normal(0, 0.05, (N, 2))
            x0 = (best_points + noise).flatten()
            
            def objective(x):
                points = x.reshape(N, 2)
                return -_score(points)
                
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                refined_points = result.x.reshape(N, 2)
                current_score = _score(refined_points)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = refined_points.copy()
                    
            except Exception:
                pass

    # Fallback if something went wrong
    if best_points is None:
        best_points = np.random.rand(N, 2)
        
    return best_points
# EVOLVE-BLOCK-END
