import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

def soft_min_objective(flat_params, k=100.0):
    """
    Differentiable approximation of the minimum area using log-sum-exp.
    min(a_i) approx -1/k * log(sum(exp(-k * a_i)))
    """
    points = flat_params.reshape(N, 2)
    areas = triangle_areas_vectorized(points)
    
    # Handle case where all areas are 0 or very small
    # If points coincide, area is 0.
    # We want to maximize min area, so we minimize -min area.
    # If min area is 0, -min area is 0.
    # Soft min of 0s is -log(N)/k? No.
    # min(0,0,..) = 0.
    # -1/k log(sum(exp(0))) = -1/k log(N).
    # This is negative.
    # So -soft_min is positive.
    # This might bias towards separating points even if they are at 0.
    
    neg_areas = -areas
    M = np.max(neg_areas)
    if M < -1e10:
        return 1e6
    exp_vals = np.exp(neg_areas - M)
    log_sum = np.log(np.sum(exp_vals))
    soft_max_neg = M + log_sum
    soft_min = -soft_max_neg
    
    pen = penalty(points)
    return -soft_min + 1000.0 * pen

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all triangles formed by the points using vectorization.
    """
    n = points.shape[0]
    # Create indices for all unique triplets (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # Vectorized cross product magnitude
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = np.min(triangle_areas_vectorized(points))
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6
    return objective(flat_params)

def soft_min_objective(flat_params, k=100.0):
    """
    Differentiable approximation of the minimum area using log-sum-exp.
    min(a_i) approx -1/k * log(sum(exp(-k * a_i)))
    """
    points = flat_params.reshape(N, 2)
    areas = triangle_areas_vectorized(points)
    
    neg_areas = -areas
    M = np.max(neg_areas)
    if M < -1e10:
        return 1e6
    exp_vals = np.exp(neg_areas - M)
    log_sum = np.log(np.sum(exp_vals))
    soft_max_neg = M + log_sum
    soft_min = -soft_max_neg
    
    pen = penalty(points)
    return -soft_min + 1000.0 * pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        min_area = np.min(triangle_areas_vectorized(points))
        pen = penalty(points)
        if pen < 1e-4:
            score = min_area
        else:
            score = min_area - 1000 * pen
            
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate Initial Guesses
    inits = []
    
    # Random restarts
    for seed in range(50):
        if time.time() - start_time > time_limit - 30:
            break
        rng = np.random.RandomState(seed)
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
        inits.append(init_points)
        
    # Add a structured symmetric init
    # 1 center, 3 near vertices, 3 near mid-edges, 3 inner ring
    sym_pts = np.zeros((N, 2))
    # Center
    sym_pts[0] = [0.5, SQRT3/6]
    # Near vertices (1/5 from vertex along median)
    # V1 (0,0). Median to (0.75, SQRT3/4). Point: (0.15, SQRT3/20)
    sym_pts[1] = [0.15, SQRT3/20]
    # V2 (1,0). Median to (0.25, SQRT3/4). Point: (0.85, SQRT3/20)
    sym_pts[2] = [0.85, SQRT3/20]
    # V3 (0.5, SQRT3/2). Median to (0.5, 0). Point: (0.5, SQRT3/2 * 0.8)
    sym_pts[3] = [0.5, SQRT3/2 * 0.8]
    
    # Near mid-edges
    # Edge 1 (0,0)-(1,0). Mid (0.5,0). Point: (0.5, SQRT3/20)
    sym_pts[4] = [0.5, SQRT3/20]
    # Edge 2 (1,0)-(0.5, SQRT3/2). Mid (0.75, SQRT3/4). Point: (0.8, SQRT3/8)
    sym_pts[5] = [0.8, SQRT3/8]
    # Edge 3 (0.5, SQRT3/2)-(0,0). Mid (0.25, SQRT3/4). Point: (0.2, SQRT3/8)
    sym_pts[6] = [0.2, SQRT3/8]
    
    # Inner ring
    # Between center and vertices
    sym_pts[7] = [0.35, SQRT3/12]
    sym_pts[8] = [0.65, SQRT3/12]
    sym_pts[9] = [0.5, SQRT3/6 + SQRT3/12]
    sym_pts[10] = [0.5, SQRT3/6 - SQRT3/12]
    
    inits.append(sym_pts)

    for init in inits:
        if time.time() - start_time > time_limit - 10:
            break
            
        flat_init = init.flatten()
        
        try:
            # Phase 1: L-BFGS-B with soft min
            res_lbgfs = minimize(
                soft_min_objective, 
                flat_init, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'maxfev': 1000, 'disp': False}
            )
            update_best(res_lbgfs.x.reshape(N, 2))
            
            # Phase 2: Powell refinement on raw objective
            res_powell = minimize(
                safe_objective, 
                res_lbgfs.x, 
                method='Powell', 
                options={'maxiter': 100, 'maxfev': 5000, 'disp': False}
            )
            update_best(res_powell.x.reshape(N, 2))
            
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            bh_result = basinhopping(
                safe_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 50, 'maxfev': 2000},
                niter=10,
                seed=42
            )
            update_best(bh_result.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
