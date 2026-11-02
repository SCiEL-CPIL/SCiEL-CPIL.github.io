import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
MAX_TIME = 115.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / hull area."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    ha = _hull_area_fast(points)
    if ha < 1e-9:
        return 0.0
        
    ma = _min_triangle_area_vec(points)
    if ma < 1e-9:
        return 0.0
        
    return ma / ha

def _decode_central_symmetric(params):
    """
    Decode parameters for a centrally symmetric 14-gon.
    params: array of shape (7, 2) -> 7 points in the first half.
    The full set is [P0, P1, ..., P6, -P0, -P1, ..., -P6].
    We must ensure the points form a convex polygon in counter-clockwise order.
    Strategy:
    1. Take the 7 points.
    2. Calculate angles relative to origin.
    3. Sort them by angle.
    4. Construct the full 14 points by appending negatives (which are +pi away).
    5. Ensure the full set is sorted by angle to form the convex hull order.
    """
    P_half = params.reshape(7, 2)
    
    # Ensure no point is at origin (degenerate)
    norms = np.linalg.norm(P_half, axis=1)
    if np.any(norms < 1e-6):
        return None
        
    # Angles of the 7 points
    angles = np.arctan2(P_half[:, 1], P_half[:, 0])
    
    # Sort the 7 points by angle
    sort_idx = np.argsort(angles)
    P_half_sorted = P_half[sort_idx]
    angles_sorted = angles[sort_idx]
    
    # Construct full 14 points
    # The negatives will have angles angle + pi
    # We need to interleave them correctly or just sort the whole set.
    # Let's create the full set and sort by angle.
    
    P_full = np.vstack([P_half_sorted, -P_half_sorted])
    angles_full = np.arctan2(P_full[:, 1], P_full[:, 0])
    
    # Sort full set by angle
    full_sort_idx = np.argsort(angles_full)
    P_final = P_full[full_sort_idx]
    
    # Check if the polygon is convex (cross products of consecutive edges should have same sign)
    # For a convex polygon in CCW order, cross product of (P[i+1]-P[i]) and (P[i+2]-P[i+1]) should be positive.
    # Actually, simpler: check if the sorted points form a convex hull with all 14 vertices.
    # If any point is inside the hull of the others, it's not a strictly convex 14-gon.
    # ConvexHull will return fewer points if some are interior.
    
    try:
        hull = ConvexHull(P_final)
        if len(hull.vertices) < N:
            # Not all points are on the hull, so it's not a strictly convex 14-gon configuration
            # We penalize this in the objective, or return None if we want strict.
            # For DE, returning a low score is better than crashing.
            pass
    except Exception:
        return None
        
    return P_final

def _objective_de(params):
    """
    Objective for Differential Evolution.
    Minimizes hull_area / min_triangle_area.
    """
    pts = _decode_central_symmetric(params)
    if pts is None:
        return 1e10
        
    ha = _hull_area_fast(pts)
    if ha < 1e-9:
        return 1e10
        
    ma = _min_triangle_area_vec(pts)
    if ma < 1e-9:
        return 1e10
        
    # Penalty if not strictly convex 14-gon
    try:
        hull = ConvexHull(pts)
        if len(hull.vertices) < N:
            return 1e10 # Or a large penalty proportional to how many are missing
    except Exception:
        return 1e10
        
    return ha / ma

def _regular_polygon_params():
    """
    Generate parameters for a regular 14-gon.
    The 7 points are at angles 0, 2pi/7, ..., 12pi/7 with radius 1.
    """
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    P_half = np.column_stack([np.cos(angles), np.sin(angles)])
    return P_half.flatten()

def construct():
    t0 = time.time()
    deadline = t0 + MAX_TIME
    
    # Baseline: Regular 14-gon
    base_params = _regular_polygon_params()
    best_pts = _decode_central_symmetric(base_params)
    if best_pts is None:
        # Fallback to simple regular polygon if decode fails (shouldn't happen)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_pts)
    
    # Strategy:
    # 1. Global search using Differential Evolution on the 14 parameters (7 points).
    # 2. Local refinement using Nelder-Mead.
    
    # Bounds for DE: Radius between 0.5 and 2.0, angle effectively handled by coordinates.
    # Since we sort by angle, the specific order of input doesn't matter, but the coordinates do.
    # We can bound x, y in [-2, 2].
    bounds = [(-2.0, 2.0)] * 14
    
    # DE is stochastic, use a fixed seed for reproducibility in this block if needed,
    # but DE handles its own randomness.
    
    try:
        # Use a moderate population size and maxiter to fit in time budget.
        # 14 dimensions.
        result_de = differential_evolution(
            _objective_de,
            bounds,
            x0=base_params,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False # We will do our own polish
        )
        
        if time.time() < deadline - 10:
            if result_de.success or result_de.fun < 1e9:
                candidate_params = result_de.x
                candidate_pts = _decode_central_symmetric(candidate_params)
                if candidate_pts is not None:
                    s = _score(candidate_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = candidate_pts
    except Exception:
        pass

    # Local Refinement: Nelder-Mead on the 14 parameters
    if time.time() < deadline - 5:
        try:
            # Start from the best params found so far
            # We need to map best_pts back to params? 
            # It's easier to just restart NM from the regular polygon or the DE result.
            # Let's use the DE result if it was valid, else regular.
            start_params = result_de.x if (result_de is not None and result_de.fun < 1e9) else base_params
            
            res_nm = minimize(
                _objective_de,
                start_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res_nm.fun < 1e9:
                candidate_pts = _decode_central_symmetric(res_nm.x)
                if candidate_pts is not None:
                    s = _score(candidate_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = candidate_pts
        except Exception:
            pass

    # Ensure we return a valid configuration
    if best_pts is None or _score(best_pts) < 1e-6:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Normalize to unit area
    ha = _hull_area_fast(best_pts)
    if ha > 1e-9:
        best_pts = best_pts * np.sqrt(1.0 / ha)
        
    return best_pts

# EVOLVE-BLOCK-END
