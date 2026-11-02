"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triplets
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 145  # Safety margin for 150s limit

def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])

def _min_triangle_area(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5

def _soft_min_area(pts, beta):
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    # Phase 1: Global Search with Differential Evolution
    if _elapsed_ok(60):
        def de_obj(flat):
            pts = flat.reshape(-1, 2)
            # Simple penalty for out of bounds, though DE bounds handle it
            return -_min_triangle_area(_project_to_triangle(pts))

        bounds = [(0, 1), (0, S3H)] * N
        try:
            de_res = differential_evolution(
                de_obj, bounds, maxiter=60, popsize=20,
                seed=123, tol=1e-8, polish=False,
                init='latinhypercube', mutation=(0.5, 1.0), recombination=0.8
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase 2: Local Refinement using Active Set Identification + SLSQP
    # This is the key improvement: instead of soft-min, we identify the active
    # triplets (those near the minimum) and optimize the min of that subset.
    
    if best_pts is not None and _elapsed_ok(20):
        current_pts = best_pts.copy()
        
        # Multiple restarts from the best point found so far
        for restart in range(5):
            if not _elapsed_ok(10):
                break
            
            # Perturb slightly to escape potential local minima
            if restart > 0:
                current_pts = _project_to_triangle(current_pts + rng.normal(0, 0.01, size=(N, 2)))
            
            # Iterative Active Set Optimization
            for iter_count in range(10):
                if not _elapsed_ok(5):
                    break
                
                areas = _all_triangle_areas(current_pts)
                min_area = np.min(areas)
                
                # Identify active triplets: those with area < min_area + tolerance
                # Tolerance is relative to the range of areas
                area_range = np.max(areas) - np.min(areas)
                tol = 0.05 * area_range + 1e-6
                active_mask = areas < (min_area + tol)
                active_indices = np.where(active_mask)[0]
                
                if len(active_indices) == 0:
                    active_indices = np.array([np.argmin(areas)])
                
                # Define objective: maximize the minimum area among active triplets
                # We minimize -min(areas[active])
                def active_obj(flat):
                    pts = flat.reshape(-1, 2)
                    a = _all_triangle_areas(pts)
                    return -np.min(a[active_indices])
                
                # Constraints: points must be in triangle
                # SLSQP works better with explicit constraints or bounds
                # We use bounds for x,y and rely on projection if needed, 
                # but SLSQP doesn't project. We must use constraints for triangle.
                
                # Triangle constraints:
                # 1. y >= 0
                # 2. y <= S3H
                # 3. x >= y/S3
                # 4. x <= 1 - y/S3
                
                constraints = []
                for i in range(N):
                    # y >= 0
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
                    # y <= S3H
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: S3H - x[2*i+1]})
                    # x >= y/S3  => x - y/S3 >= 0
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*i+1]/S3})
                    # x <= 1 - y/S3 => 1 - y/S3 - x >= 0
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1]/S3 - x[2*i]})

                x0 = current_pts.flatten()
                
                try:
                    res = minimize(
                        active_obj, 
                        x0, 
                        method='SLSQP', 
                        bounds=[(0, 1), (0, S3H)] * N,
                        constraints=constraints,
                        options={'maxiter': 200, 'ftol': 1e-12}
                    )
                    if res.success:
                        new_pts = _project_to_triangle(res.x.reshape(-1, 2))
                        new_area = _min_triangle_area(new_pts)
                        if new_area > min_area - 1e-9:
                            current_pts = new_pts
                            _update_best(new_pts)
                        else:
                            break # No improvement
                    else:
                        break
                except Exception:
                    break

    # Phase 3: Final High-Precision Polish
    if best_pts is not None and _elapsed_ok(5):
        # Use Nelder-Mead for a final touch-up without constraints (relying on projection)
        # This can sometimes find slightly better local minima than SLSQP due to different path
        def nm_obj(flat):
            pts = flat.reshape(-1, 2)
            return -_min_triangle_area(_project_to_triangle(pts))
        
        x0 = best_pts.flatten()
        try:
            res = minimize(
                nm_obj, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.success:
                new_pts = _project_to_triangle(res.x.reshape(-1, 2))
                _update_best(new_pts)
        except Exception:
            pass

    if best_pts is None:
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [0.333, 0.0],
            [0.667, 0.0],
            [0.5, S3H/3],
            [0.2, 0.1],
        ], dtype=float)
        return _project_to_triangle(pts)

    return _project_to_triangle(best_pts)
