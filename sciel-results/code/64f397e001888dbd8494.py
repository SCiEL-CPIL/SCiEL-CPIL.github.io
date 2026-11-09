import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2
CENTROID = np.array([0.5, H / 3.0])

# Precompute triangle indices for the full 11-point set
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
    Reconstruct the full 11x2 points array from the reduced 12x1 parameters.
    reduced_params: [cx, cy, x1, y1, ..., x5, y5]
    """
    cx, cy = reduced_params[0], reduced_params[1]
    pts = np.empty((11, 2))
    pts[0] = [cx, cy]
    
    right_pts = reduced_params[2:].reshape(5, 2)
    # Ensure right points are in the right half (x >= 0.5) to avoid symmetry ambiguity in search
    # Actually, we can let the optimizer decide, but constraining x >= 0.5 helps.
    # For now, we just mirror them.
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

def is_inside_full(points):
    """
    Check if all points are inside the equilateral triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    if np.any(y < -1e-9):
        return False
    # y <= sqrt(3) * x
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    # y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
    return True

def objective_reduced(params):
    """
    Objective function for the reduced 12D space.
    Returns -min_area if valid, penalty if invalid.
    """
    points = get_full_points(params)
    
    if not is_inside_full(points):
        # Penalty should be larger than any possible valid score (which is < 0.1)
        return 1.0
        
    area = min_triangle_area_vec(points)
    return -area

def refine_local_reduced(params, deadline):
    """
    Perform local refinement on the reduced parameter vector.
    """
    if time.time() > deadline:
        return params
    
    best_params = params.copy()
    best_score = -objective_reduced(best_params)
    
    rng = np.random.default_rng(42)
    
    # Try Nelder-Mead with several perturbations
    for attempt in range(15):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            x0 = best_params.copy()
        else:
            # Small perturbation
            perturbation = rng.normal(0, 0.01, size=best_params.shape)
            x0 = best_params + perturbation
            # Ensure validity roughly by checking the full points
            if not is_inside_full(get_full_points(x0)):
                continue
        
        try:
            res = minimize(
                objective_reduced,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
            )
            
            new_params = res.x
            if is_inside_full(get_full_points(new_params)):
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params.copy()
                    
        except Exception:
            continue
            
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Reduced dimensions:
    # cx, cy
    # 5 * (x, y) for the right half points
    # Total 12 variables
    
    # Bounds for cx, cy: inside the medial triangle (to keep centroid somewhat central)
    # Actually, centroid can be anywhere in the triangle, but optimal is likely central.
    # Let's use the full triangle bounds for safety, but narrow for efficiency.
    # For the 5 right points: x in [0.5, 1.0], y in [0, H]
    
    bounds = [
        (0.1, 0.9),  # cx
        (0.1, H - 0.1), # cy
    ]
    for _ in range(5):
        bounds.append((0.5, 1.0)) # x_right
        bounds.append((0.0, H))   # y_right
        
    best_params = None
    best_score = -float('inf')
    
    # Differential Evolution on reduced space
    try:
        # popsize 20, maxiter 300 for 12D space is feasible
        result = differential_evolution(
            objective_reduced,
            bounds,
            maxiter=300,
            popsize=20,
            tol=0.001,
            seed=42,
            polish=False,  # We do our own polish
            init='sobol'
        )
        
        candidate_params = result.x
        if is_inside_full(get_full_points(candidate_params)):
            candidate_score = -result.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params.copy()
                
    except Exception:
        pass
        
    # Fallback if DE failed
    if best_params is None:
        # Construct a reasonable symmetric guess
        # Centroid
        cx, cy = 0.5, H/3.0
        # 5 points on a circle? Or regular grid in the right half?
        # Let's just use random valid points in the right half
        rng = np.random.default_rng(0)
        for _ in range(100):
            pts_right = rng.random((5, 2))
            pts_right[:, 0] = 0.5 + pts_right[:, 0] * 0.5 # x in [0.5, 1.0]
            pts_right[:, 1] = pts_right[:, 1] * H         # y in [0, H]
            
            params = np.concatenate([[cx, cy], pts_right.flatten()])
            if is_inside_full(get_full_points(params)):
                best_params = params
                best_score = -objective_reduced(params)
                break
        if best_params is None:
            # Last resort
            best_params = np.array([0.5, H/3.0] + [0.5, H/6.0]*5)
            best_score = -objective_reduced(best_params)
    
    # Local Refinement
    if time.time() < deadline - 10:
        best_params = refine_local_reduced(best_params, deadline)
        
    # Convert back to full 11x2 points
    final_points = get_full_points(best_params)
    
    return final_points
