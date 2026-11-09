# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, differential_evolution
import itertools

_DEADLINE = time.time() + 615.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _triangle_areas(points):
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return areas

def _soft_min_objective(pts_flat, alpha=50.0):
    """Differentiable soft-min surrogate for min triangle area / hull area."""
    pts = pts_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        
        n = len(pts)
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Soft minimum using log-sum-exp
        min_area = np.min(areas)
        if min_area < 1e-10:
            return 1e6
        
        # Normalize for numerical stability
        shifted = areas - min_area
        log_sum_exp = np.log(np.sum(np.exp(-alpha * shifted))) / alpha
        soft_min_area = min_area - log_sum_exp
        
        if soft_min_area <= 0:
            return 1e6
        
        return -(soft_min_area / ha)
    except Exception:
        return 1e6

def _concentric_ring_config(n=13, n_rings=3, base_radius=0.45, ring_spacing=0.15, angle_offsets=None):
    """
    Create a structured concentric-ring topology.
    Distributes points across rings with angle offsets to maximize spacing.
    """
    if angle_offsets is None:
        angle_offsets = []
    
    points = []
    remaining = n
    
    for ring in range(n_rings):
        if remaining <= 0:
            break
        
        # Determine how many points on this ring
        if ring == n_rings - 1:
            n_on_ring = remaining
        else:
            # Distribute roughly evenly, with more on outer rings
            n_on_ring = max(1, int(round(remaining * (ring + 1) / (n_rings * (n_rings + 1) / 2))))
            remaining -= n_on_ring
        
        if n_on_ring <= 0:
            continue
        
        radius = base_radius - ring * ring_spacing
        if radius < 0.05:
            radius = 0.05
        
        # Angle offset for this ring
        offset = angle_offsets[ring] if ring < len(angle_offsets) else (np.pi / (2 * (ring + 1)))
        
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False) + offset
        ring_pts = np.column_stack([0.5 + radius * np.cos(angles),
                                     0.5 + radius * np.sin(angles)])
        points.append(ring_pts)
    
    return np.vstack(points) if points else np.array([[0.5, 0.5]] * n)

def _optimize_powell(points0, alpha=50.0, maxiter=100):
    """
    Optimize using Powell's method with soft-min surrogate.
    Powell's method is derivative-free and handles non-smooth landscapes better
    than L-BFGS-B for this problem.
    """
    pts_flat = points0.flatten()
    n = len(points0)
    bounds = [(0.01, 0.99)] * (2 * n)
    
    try:
        result = minimize(
            _soft_min_objective,
            pts_flat,
            args=(alpha,),
            method='Powell',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-10, 'maxfev': 5000}
        )
        if result.x is not None:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points0

def _exact_refinement(points0, max_sweeps=200):
    """
    Exact refinement: iteratively find the minimum-area triangle and
    move its vertices to increase the minimum area.
    """
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
            break
        
        # Find the minimum area triangle
        areas = _triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        if min_area < 1e-10:
            break
        
        # Get the three points in the critical triangle
        idx = np.array(list(itertools.combinations(range(n), 3)))
        crit_indices = idx[min_idx]
        
        improved = False
        # Try moves for each of the 3 critical points
        for pi in crit_indices:
            x0, y0 = pts[pi]
            best_x, best_y = x0, y0
            best_local = best_score
            
            # Multi-scale step sizes
            for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[pi] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local + 1e-12:
                        best_local = s
                        best_x, best_y = nx, ny
                        improved = True
            
            pts[pi] = [best_x, best_y]
        
        cur = _score(pts)
        if cur > best_score + 1e-12:
            best_score = cur
            best_pts = pts.copy()
        
        if not improved:
            break
    
    return best_pts, best_score

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _good_starting_config(n=13):
    """Create a good starting configuration with hull points on a circle and interior points spread out."""
    # 8 hull points on a circle
    n_hull = 8
    hull_angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    hull_pts = np.column_stack([0.5 + 0.45 * np.cos(hull_angles),
                                 0.5 + 0.45 * np.sin(hull_angles)])
    
    # 5 interior points in a smaller pentagon
    n_int = n - n_hull
    int_angles = np.linspace(0, 2 * np.pi, n_int, endpoint=False) + np.pi / n_int
    int_pts = np.column_stack([0.5 + 0.15 * np.cos(int_angles),
                                0.5 + 0.15 * np.sin(int_angles)])
    
    return np.vstack([hull_pts, int_pts])

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Concentric ring topology + Powell's method
    ring_configs = [
        _concentric_ring_config(n, n_rings=3, base_radius=0.45, ring_spacing=0.15,
                                angle_offsets=[0, np.pi/3, np.pi/6]),
        _concentric_ring_config(n, n_rings=3, base_radius=0.45, ring_spacing=0.15,
                                angle_offsets=[0, np.pi/4, np.pi/8]),
        _concentric_ring_config(n, n_rings=2, base_radius=0.45, ring_spacing=0.20,
                                angle_offsets=[0, np.pi/5]),
        _concentric_ring_config(n, n_rings=4, base_radius=0.45, ring_spacing=0.12,
                                angle_offsets=[0, np.pi/6, np.pi/12, np.pi/24]),
    ]
    
    for start in ring_configs:
        if time.time() > _DEADLINE - 60:
            break
        
        # Powell's method with moderate alpha
        opt_pts = _optimize_powell(start, alpha=30.0, maxiter=80)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
        
        # Powell's method with higher alpha for sharper min
        if time.time() < _DEADLINE - 40:
            opt_pts2 = _optimize_powell(opt_pts, alpha=80.0, maxiter=60)
            s2 = _score(opt_pts2)
            if s2 > best_score:
                best_score = s2
                best_points = opt_pts2.copy()
        
        # Exact refinement
        if time.time() < _DEADLINE - 20:
            refined_pts, refined_s = _exact_refinement(opt_pts, max_sweeps=150)
            if refined_s > best_score:
                best_score = refined_s
                best_points = refined_pts
    
    # Strategy 2: Traditional good starting config + Powell
    if time.time() < _DEADLINE - 80:
        start = _good_starting_config(n)
        opt_pts = _optimize_powell(start, alpha=40.0, maxiter=100)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
        
        if time.time() < _DEADLINE - 30:
            opt_pts2 = _optimize_powell(opt_pts, alpha=100.0, maxiter=60)
            s2 = _score(opt_pts2)
            if s2 > best_score:
                best_score = s2
                best_points = opt_pts2.copy()
        
        if time.time() < _DEADLINE - 20:
            refined_pts, refined_s = _exact_refinement(opt_pts, max_sweeps=200)
            if refined_s > best_score:
                best_score = refined_s
                best_points = refined_pts
    
    # Strategy 3: Regular polygon + Powell
    if time.time() < _DEADLINE - 60:
        start2 = _regular_polygon(n, radius=0.45, center=(0.5, 0.5))
        opt_pts3 = _optimize_powell(start2, alpha=50.0, maxiter=80)
        s3 = _score(opt_pts3)
        if s3 > best_score:
            best_score = s3
            best_points = opt_pts3.copy()
        
        if time.time() < _DEADLINE - 20:
            refined_pts2, refined_s2 = _exact_refinement(opt_pts3, max_sweeps=150)
            if refined_s2 > best_score:
                best_score = refined_s2
                best_points = refined_pts2
    
    # Final exact refinement
    if best_points is not None and time.time() < _DEADLINE - 15:
        final_pts, final_s = _exact_refinement(best_points, max_sweeps=300)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
