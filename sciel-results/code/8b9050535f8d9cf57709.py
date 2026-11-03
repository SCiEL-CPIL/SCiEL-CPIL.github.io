import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations of indices
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    Returns -soft_min so that we can minimize it.
    """
    areas = _compute_areas(points)
    # Shift to prevent overflow/underflow
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - log_sum / beta
    return -soft_min

def _project_to_triangle(points):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, H).
    This is a simple clipping/projection for robustness.
    """
    pts = points.copy()
    # Clip to bounding box
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, H)
    
    # Enforce triangle inequalities
    # y <= sqrt(3) * x  =>  y - sqrt(3)x <= 0
    # y <= -sqrt(3)(x - 1)  =>  y + sqrt(3)x - sqrt(3) <= 0
    
    # Iterative projection might be needed for accuracy, but simple clipping is often sufficient for initialization.
    # For optimization, we will use bounds and let the optimizer stay inside, or use a penalty.
    # However, for the final output, we must ensure validity.
    
    # Simple geometric projection for points that are outside
    for i in range(len(pts)):
        x, y = pts[i]
        
        # Check if outside
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y > H:
            y = H
            
        # Check edge y = sqrt(3)x
        if y > SQRT3 * x:
            # Project onto the line
            # Normal vector n = (-sqrt(3), 1)
            # Point p = (x, y)
            # Line passes through (0,0)
            # Projection of p onto line spanned by (1, sqrt(3)) is:
            # t = (p . v) / (v . v) where v = (1, sqrt(3))
            v = np.array([1.0, SQRT3])
            p = np.array([x, y])
            t = np.dot(p, v) / np.dot(v, v)
            if t < 0: t = 0
            if t > 1: t = 1 # Should not exceed 1 for this edge if x,y are reasonable
            proj = t * v
            # Check if this projection is valid (within segment)
            # Actually, just clipping to the half-space is safer for simple projection
            # Let's use the half-space correction
            # Distance to line y - sqrt(3)x = 0 is (y - sqrt(3)x) / 2
            # We want y - sqrt(3)x <= 0
            # If violated, move along normal (-sqrt(3), 1)
            dist = (y - SQRT3 * x) / 2.0
            x_new = x - SQRT3 * dist
            y_new = y + 1.0 * dist
            pts[i] = [x_new, y_new]
            
        # Check edge y = -sqrt(3)(x-1) => y + sqrt(3)x - sqrt(3) <= 0
        val = y + SQRT3 * x - SQRT3
        if val > 0:
            dist = val / 2.0
            # Normal is (sqrt(3), 1)
            x_new = x - SQRT3 * dist
            y_new = y - 1.0 * dist
            pts[i] = [x_new, y_new]
            
    return pts

def _is_valid(points):
    """Check if all points are inside the triangle."""
    eps = 1e-8
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    if np.any(y > SQRT3 * x + eps):
        return False
    if np.any(y > -SQRT3 * x + SQRT3 + eps):
        return False
    return True

def _objective_cartesian(points_flat, beta):
    """
    Objective function for Cartesian optimization.
    """
    points = points_flat.reshape(N, 2)
    # We assume the optimizer keeps points within bounds, but for safety:
    # points = _project_to_triangle(points)
    return _soft_min_area(points, beta)

def _local_polish(points, time_limit=5.0, rng=None):
    """
    Local search that moves points involved in the smallest triangles.
    This is a heuristic refinement step.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_points = points.copy()
    current_min = _min_area(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    start_t = time.time()
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        improved = True
        while improved and time.time() - start_t < time_limit:
            improved = False
            areas = _compute_areas(current_points)
            min_area = float(np.min(areas))
            
            if min_area > best_min + 1e-10:
                best_min = min_area
                best_points = current_points.copy()
                
            # Find points involved in min triangles
            threshold = min_area + 1e-8
            min_indices = np.where(areas <= threshold)[0]
            if len(min_indices) == 0:
                break
            
            involved = set()
            for idx in min_indices:
                involved.add(int(_idx_arr[idx, 0]))
                involved.add(int(_idx_arr[idx, 1]))
                involved.add(int(_idx_arr[idx, 2]))
            
            if len(involved) == 0:
                break
            
            # Try moving one of the involved points
            for idx in list(involved):
                if time.time() - start_t > time_limit:
                    break
                
                # Try random perturbations
                for _ in range(50):
                    du = rng.uniform(-step_size, step_size)
                    dv = rng.uniform(-step_size, step_size)
                    
                    new_x = current_points[idx, 0] + du
                    new_y = current_points[idx, 1] + dv
                    
                    # Check bounds roughly
                    if new_x < 0 or new_x > 1 or new_y < 0 or new_y > H:
                        continue
                    # Check triangle inequalities roughly
                    if new_y > SQRT3 * new_x + 1e-6:
                        continue
                    if new_y > -SQRT3 * new_x + SQRT3 + 1e-6:
                        continue
                        
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        min_area = test_min
                        improved = True
                        break
                        
                if improved:
                    break
                    
    return best_points, best_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Define bounds for differential evolution
    # x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    # Phase 1: Global Optimization with Differential Evolution
    # Use a low beta for a smooth landscape
    beta_de = 10.0
    
    try:
        de_result = differential_evolution(
            _objective_cartesian,
            bounds,
            args=(beta_de,),
            seed=42,
            maxiter=200,
            tol=1e-6,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        if de_result.success or de_result.fun < best_score:
            candidate_points = de_result.x.reshape(N, 2)
            candidate_points = _project_to_triangle(candidate_points)
            if _is_valid(candidate_points):
                score = _min_area(candidate_points)
                if score > best_score:
                    best_score = score
                    best_points = candidate_points
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_points is not None:
            return best_points
        else:
            # Fallback to random
            pts = rng.random((N, 2)) * [1.0, H]
            pts = _project_to_triangle(pts)
            return pts

    # Phase 2: Local Refinement with L-BFGS-B and increasing Beta (Annealing)
    if best_points is not None:
        current_points = best_points.copy()
        
        # Schedule of betas
        betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        
        for beta in betas:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    _objective_cartesian,
                    current_points.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < best_score: # Note: res.fun is negative soft min
                    candidate_points = res.x.reshape(N, 2)
                    candidate_points = _project_to_triangle(candidate_points)
                    if _is_valid(candidate_points):
                        score = _min_area(candidate_points)
                        if score > best_score:
                            best_score = score
                            best_points = candidate_points
                            current_points = candidate_points
            except Exception:
                pass
    
    # Phase 3: Heuristic Local Polish
    if best_points is not None and time.time() < deadline - 10:
        polished_points, polished_score = _local_polish(best_points, time_limit=10.0, rng=rng)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points

    # Phase 4: One more L-BFGS-B pass with very high beta if time permits
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_cartesian,
                best_points.flatten(),
                args=(2000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            candidate_points = res.x.reshape(N, 2)
            candidate_points = _project_to_triangle(candidate_points)
            if _is_valid(candidate_points):
                score = _min_area(candidate_points)
                if score > best_score:
                    best_score = score
                    best_points = candidate_points
        except Exception:
            pass

    # Final Validation
    if best_points is None:
        # Generate a valid random configuration as a last resort
        pts = rng.random((N, 2)) * [1.0, H]
        best_points = _project_to_triangle(pts)
    else:
        best_points = _project_to_triangle(best_points)
        
    return best_points
