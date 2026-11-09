import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    i, j, k = i.T.ravel(), j.T.ravel(), k.T.ravel()
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Return the minimum triangle area, or 0 if degenerate."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        areas = _triangle_areas(points)
        return float(np.min(areas))
    except Exception:
        return 0.0

def _objective(points_flat):
    """Objective function for minimization: -min_triangle_area.
    We minimize the negative of the minimum area to maximize it.
    We also add a small penalty if points go too far out to keep the hull manageable,
    though the ratio handles scale invariance, fixing scale helps optimization.
    """
    points = points_flat.reshape(13, 2)
    # Optional: normalize to unit circle to help convergence
    # But for Nelder-Mead, raw coordinates are fine if starts are good.
    min_area = _min_triangle_area(points)
    if min_area == 0.0:
        return 1e6 # Large penalty for degenerate triangles
    return -min_area

def _generate_initial_points(rng, strategy):
    """Generate diverse initial point configurations."""
    n = 13
    if strategy == 0:
        # Regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    elif strategy == 1:
        # Random points in unit disk
        r = np.sqrt(rng.random(n))
        theta = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    elif strategy == 2:
        # Two concentric rings
        n1 = 7
        n2 = 6
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        r1 = 1.0
        r2 = 0.5
        pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        return np.vstack([pts1, pts2])
    elif strategy == 3:
        # Jittered grid
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack((xx.ravel(), yy.ravel()))[:13]
        pts += rng.normal(0, 0.1, size=pts.shape)
        return pts
    else:
        # Random with some structure
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0.5, 1.0, n)
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate a pool of diverse initial configurations
    initial_configs = []
    for s in range(5):
        for _ in range(3): # 3 variants per strategy
            initial_configs.append(_generate_initial_points(rng, s))
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Multi-start Nelder-Mead
    # We will run Nelder-Mead from each initial config.
    # To manage time, we limit the number of function evaluations per start.
    max_fevals_per_start = 2000
    
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break
            
        # Flatten points for optimizer
        x0 = init_pts.ravel()
        
        # Define options for Nelder-Mead
        options = {
            'maxiter': 1000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'maxfev': max_fevals_per_start
        }
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options=options)
            if res.success:
                candidate_pts = res.x.reshape(13, 2)
                score = _score(candidate_pts)
                if score > best_score:
                    best_score = score
                    best_points = candidate_pts.copy()
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 30:
            break

    # Phase 2: Local Refinement (Coordinate Descent) on the best candidate
    if best_points is not None and time.time() < deadline - 10:
        # Implement a fast coordinate descent to polish
        pts = best_points.copy()
        n = 13
        # Try moving each point slightly in x and y to improve min area
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                for coord in range(2):
                    # Simple line search: try a few steps
                    base_val = _min_triangle_area(pts)
                    best_val = base_val
                    best_step = 0.0
                    
                    # Try positive and negative steps with decreasing size
                    for step_size in [0.1, 0.05, 0.01, 0.005]:
                        for sign in [1, -1]:
                            new_pts = pts.copy()
                            new_pts[i, coord] += sign * step_size
                            val = _min_triangle_area(new_pts)
                            if val > best_val:
                                best_val = val
                                best_step = sign * step_size
                                best_pts_temp = new_pts.copy()
                    
                    if best_step != 0.0 and best_val > base_val:
                        pts = best_pts_temp
                        improved = True
                        score = _score(pts)
                        if score > best_score:
                            best_score = score
                            best_points = pts.copy()

    # Fallback if no valid points found
    if best_points is None or _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
