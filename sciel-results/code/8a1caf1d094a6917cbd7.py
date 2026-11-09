import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def get_full_points(reduced_params):
    """
    Reconstruct 11x2 points from 11 reduced parameters.
    reduced_params: [cy, x1, y1, ..., x5, y5]
    Structure:
    - Point 0: (0.5, cy)
    - Points 1-5: (x_i, y_i) in right half (x >= 0.5)
    - Points 6-10: (1-x_i, y_i) in left half (x <= 0.5)
    """
    cy = reduced_params[0]
    pts = np.empty((11, 2))
    pts[0] = [0.5, cy]
    
    right_pts = reduced_params[1:].reshape(5, 2)
    pts[1:6] = right_pts
    pts[6:11] = np.column_stack([1.0 - right_pts[:, 0], right_pts[:, 1]])
    
    return pts

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area_vec(points, beta=50.0):
    """
    Smooth approximation of min area using log-sum-exp.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # soft min: -log(sum(exp(-beta * areas))) / beta
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    return -log_sum / beta

def is_inside_reduced(reduced_params):
    """
    Check if the reduced points correspond to a valid configuration inside the triangle.
    """
    cy = reduced_params[0]
    if cy < -1e-9 or cy > H + 1e-9:
        return False
        
    right_pts = reduced_params[1:].reshape(5, 2)
    x = right_pts[:, 0]
    y = right_pts[:, 1]
    
    if np.any(x < 0.5 - 1e-9):
        return False
    if np.any(y < -1e-9):
        return False
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective_soft(params, beta=50.0):
    """
    Objective for DE using soft-min surrogate.
    """
    if not is_inside_reduced(params):
        return 1.0
        
    points = get_full_points(params)
    area = soft_min_area_vec(points, beta)
    return -area

def objective_exact(params):
    """
    Objective for local refinement using exact min.
    """
    if not is_inside_reduced(params):
        return 1.0
        
    points = get_full_points(params)
    area = min_triangle_area_vec(points)
    return -area

def refine_local(reduced_params, start_time, deadline):
    """
    Perform local refinement using Nelder-Mead on the reduced parameter space.
    """
    if time.time() > deadline:
        return reduced_params
    
    best_params = reduced_params.copy()
    best_score = -objective_exact(best_params)
    
    rng = np.random.default_rng(42)
    
    for attempt in range(15):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            x0 = best_params.copy()
        else:
            perturbation = rng.normal(0, 0.005, size=best_params.shape)
            x0 = best_params + perturbation
            
            if not is_inside_reduced(x0):
                continue
        
        try:
            res = minimize(
                objective_exact,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if time.time() > deadline:
                break
                
            new_params = res.x
            if is_inside_reduced(new_params):
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params.copy()
                    
        except Exception:
            continue
            
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    bounds = [(0.0, H)]
    for i in range(5):
        bounds.append((0.5, 1.0))
        bounds.append((0.0, H))
        
    best_params = None
    best_score = -float('inf')
    
    # Phase 1: DE with soft-min surrogate
    try:
        result = differential_evolution(
            objective_soft,
            bounds,
            maxiter=500,
            popsize=25,
            tol=0.005,
            seed=42,
            polish=False,
            init='sobol',
            update_interval=1
        )
        
        candidate_params = result.x
        if is_inside_reduced(candidate_params):
            candidate_score = -objective_exact(candidate_params)
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params.copy()
                
    except Exception:
        pass
        
    if best_params is None:
        rng = np.random.default_rng(42)
        for _ in range(200):
            cy = rng.uniform(0, H)
            xs = rng.uniform(0.5, 1.0, 5)
            max_ys = np.sqrt(3) * (1 - xs)
            ys = rng.uniform(0, max_ys)
            params = np.concatenate([[cy], np.column_stack([xs, ys]).flatten()])
            if is_inside_reduced(params):
                best_params = params
                best_score = -objective_exact(best_params)
                break
        if best_params is None:
            best_params = np.array([H/3] + [0.5, 0.0]*5)
            best_params[1] = 0.5
            best_params[3] = 0.0
            if not is_inside_reduced(best_params):
                best_params = np.zeros(11)
                best_params[0] = H/3
                for i in range(5):
                    best_params[1+2*i] = 0.5
                    best_params[2+2*i] = 0.0

    # Phase 2: Local refinement with exact objective
    if time.time() < deadline - 5:
        best_params = refine_local(best_params, start_time, deadline)
        
    return get_full_points(best_params)
