import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
# i < j < k
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
    # Ensure x is in [0.5, 1]
    # The bounds in DE will handle this, but we can clip for safety in evaluation
    # However, for the objective function, we want to penalize invalid points.
    # So we just construct them as is.
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
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

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
    
    # Right half constraints:
    # 1. x >= 0.5
    if np.any(x < 0.5 - 1e-9):
        return False
    # 2. y >= 0
    if np.any(y < -1e-9):
        return False
    # 3. y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective_reduced(params):
    """
    Objective function for minimization on reduced parameters.
    Returns -min_area if valid, penalty if invalid.
    """
    if not is_inside_reduced(params):
        return 1.0  # Penalty
        
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
    best_score = -objective_reduced(best_params) # objective returns -area, so -obj is area
    
    rng = np.random.default_rng(42)
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            x0 = best_params.copy()
        else:
            # Small random perturbation
            perturbation = rng.normal(0, 0.01, size=best_params.shape)
            x0 = best_params + perturbation
            
            # Rough validity check
            if not is_inside_reduced(x0):
                continue
        
        try:
            res = minimize(
                objective_reduced,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
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
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Bounds for reduced parameters:
    # cy: [0, H]
    # x_i: [0.5, 1]
    # y_i: [0, H] (will be constrained by is_inside)
    
    bounds = [(0.0, H)]  # cy
    for i in range(5):
        bounds.append((0.5, 1.0))  # x_i
        bounds.append((0.0, H))    # y_i
        
    best_params = None
    best_score = -float('inf')
    
    # Use Differential Evolution for global search on reduced space
    try:
        result = differential_evolution(
            objective_reduced,
            bounds,
            maxiter=300,  # More iterations possible due to lower dimension
            popsize=20,
            tol=0.01,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        candidate_params = result.x
        if is_inside_reduced(candidate_params):
            candidate_score = -result.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params.copy()
                
    except Exception:
        pass
        
    if best_params is None:
        # Fallback: random valid reduced params
        rng = np.random.default_rng(42)
        for _ in range(100):
            cy = rng.uniform(0, H)
            xs = rng.uniform(0.5, 1.0, 5)
            # Max y for given x is sqrt(3)*(1-x)
            max_ys = np.sqrt(3) * (1 - xs)
            ys = rng.uniform(0, max_ys)
            params = np.concatenate([[cy], np.column_stack([xs, ys]).flatten()])
            if is_inside_reduced(params):
                best_params = params
                best_score = -objective_reduced(best_params)
                break
        if best_params is None:
            # Last resort
            best_params = np.array([H/3] + [0.5, 0.0]*5) # Degenerate, but valid structure
            best_params[1] = 0.5
            best_params[3] = 0.0
            # Ensure it's inside
            if not is_inside_reduced(best_params):
                 best_params = np.zeros(11)
                 best_params[0] = H/3
                 for i in range(5):
                     best_params[1+2*i] = 0.5
                     best_params[2+2*i] = 0.0

    # Local refinement
    if time.time() < deadline - 10:
        best_params = refine_local(best_params, start_time, deadline)
        
    return get_full_points(best_params)
