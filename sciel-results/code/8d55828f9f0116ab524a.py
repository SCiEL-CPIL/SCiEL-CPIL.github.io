import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all triangles
TRI_INDICES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRI_INDICES.append((i, j, k))
TRI_INDICES = np.array(TRI_INDICES)

def compute_areas(points):
    """
    Compute areas of all 165 triangles defined by TRI_INDICES.
    points: (N, 2)
    returns: (165,) array of areas
    """
    # points[:, 0] is x, points[:, 1] is y
    # Area = 0.5 * | x1(y2-y3) + x2(y3-y1) + x3(y1-y2) |
    # Using vectorized operations
    i, j, k = TRI_INDICES.T
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Cross product magnitude in 2D: x1*y2 - y1*x2 + x2*y3 - y2*x3 + x3*y1 - y3*x1
    # Or determinant form
    # det = (p2 - p1) x (p3 - p1)
    # dx1 = p2[0] - p1[0], dy1 = p2[1] - p1[1]
    # dx2 = p3[0] - p1[0], dy2 = p3[1] - p1[1]
    # area = 0.5 * | dx1*dy2 - dy1*dx2 |
    
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def boundary_penalty(points):
    """
    Penalty for points outside the equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Lines:
    1. y >= 0
    2. y <= sqrt(3) * x
    3. y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 * (1 - x) - y
    
    # Sum of squared violations
    v1 = np.maximum(0, -d1)
    v2 = np.maximum(0, -d2)
    v3 = np.maximum(0, -d3)
    
    return np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2)

def penalized_objective(params, lambda_area, lambda_bound):
    """
    params: flattened array of 2*N points + 1 alpha
    """
    points = params[:2*N].reshape(N, 2)
    alpha = params[2*N]
    
    areas = compute_areas(points)
    
    # Violation: alpha - area > 0
    # We want area >= alpha
    violations = np.maximum(0, alpha - areas)
    area_pen = np.sum(violations**2)
    
    bound_pen = boundary_penalty(points)
    
    # Objective: maximize alpha, minimize penalties
    # Minimize: -alpha + lambda_area * area_pen + lambda_bound * bound_pen
    return -alpha + lambda_area * area_pen + lambda_bound * bound_pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_alpha = -np.inf
    
    # Initial penalty weights
    lambda_area_start = 100.0
    lambda_bound_start = 1000.0
    
    # Try multiple random restarts
    for seed in range(30):
        if time.time() - start_time > time_limit - 20: # Leave time for refinement
            break
        
        rng = np.random.RandomState(seed)
        
        # Random initialization inside triangle
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
            
        # Initial alpha: estimate min area
        init_areas = compute_areas(init_points)
        init_alpha = np.min(init_areas)
        
        init_params = np.concatenate([init_points.flatten(), [init_alpha]])
        
        try:
            # Phase 1: Moderate penalty to find a good structure
            res1 = minimize(
                penalized_objective,
                init_params,
                args=(lambda_area_start, lambda_bound_start),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res1.success or res1.fun < 1e5: # Check for reasonable cost
                points1 = res1.x[:2*N].reshape(N, 2)
                alpha1 = res1.x[2*N]
                
                # Check if feasible (mostly)
                areas1 = compute_areas(points1)
                min_area1 = np.min(areas1)
                bound_pen1 = boundary_penalty(points1)
                
                # If the solution is feasible (alpha close to min area and points inside)
                if bound_pen1 < 1e-4 and abs(alpha1 - min_area1) < 1e-3:
                    if min_area1 > best_alpha:
                        best_alpha = min_area1
                        best_points = points1.copy()
                        
        except Exception:
            pass
            
    # Refinement phase on the best solution found
    if best_points is not None:
        if time.time() - start_time < time_limit - 5:
            # Increase penalties to force feasibility and tightness
            lambda_area_refine = 10000.0
            lambda_bound_refine = 100000.0
            
            init_params_refine = np.concatenate([best_points.flatten(), [best_alpha]])
            
            try:
                res2 = minimize(
                    penalized_objective,
                    init_params_refine,
                    args=(lambda_area_refine, lambda_bound_refine),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                
                points2 = res2.x[:2*N].reshape(N, 2)
                alpha2 = res2.x[2*N]
                
                areas2 = compute_areas(points2)
                min_area2 = np.min(areas2)
                bound_pen2 = boundary_penalty(points2)
                
                if bound_pen2 < 1e-4:
                    if min_area2 > best_alpha:
                        best_alpha = min_area2
                        best_points = points2.copy()
                        
            except Exception:
                pass

    if best_points is None:
        # Fallback: center point
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
