# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 625.0

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

def _soft_min_triangle_area(pts_flat, alpha=100.0):
    """Smooth approximation of minimum triangle area using log-sum-exp."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    # Soft minimum via log-sum-exp
    neg_areas = -alpha * areas
    max_neg = np.max(neg_areas)
    log_sum_exp = np.log(np.sum(np.exp(neg_areas - max_neg))) + max_neg
    soft_min = -log_sum_exp / alpha
    return float(soft_min)

def _soft_score_objective(pts_flat, alpha=100.0):
    """Objective: minimize negative soft-min ratio."""
    try:
        pts = pts_flat.reshape(-1, 2)
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        sm = _soft_min_triangle_area(pts_flat, alpha)
        if sm <= 0:
            return 1e6
        return -(sm / ha)
    except Exception:
        return 1e6

def _lloyd_relax_interior(points, n_iter=50, step_size=0.5):
    """
    Lloyd's algorithm: move interior points toward Voronoi cell centroids.
    Hull points stay fixed. This enforces uniform local density.
    """
    pts = points.copy()
    n = len(pts)
    
    # Identify hull vertices
    try:
        hull = ConvexHull(pts)
        hull_set = set(hull.vertices)
    except Exception:
        return pts
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(pts)
        except Exception:
            break
        
        # For each interior point, compute Voronoi centroid
        moved = False
        for i in range(n):
            if i in hull_set:
                continue
            
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded or degenerate - use Delaunay neighbor centroid
                try:
                    tri = Delaunay(pts)
                    mask = np.any(tri.simplices == i, axis=1)
                    if np.any(mask):
                        neighbors = np.unique(tri.simplices[mask].reshape(-1))
                        neighbors = neighbors[neighbors != i]
                        if len(neighbors) > 0:
                            centroid = np.mean(pts[neighbors], axis=0)
                            pts[i] = pts[i] + step_size * (centroid - pts[i])
                            moved = True
                except Exception:
                    pass
                continue
            
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                continue
            
            # Compute polygon centroid using shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            x2 = np.roll(x, -1)
            y2 = np.roll(y, -1)
            cross = x * y2 - x2 * y
            area = 0.5 * np.sum(cross)
            
            if abs(area) < 1e-12:
                continue
            
            cx = np.sum((x + x2) * cross) / (6.0 * area)
            cy = np.sum((y + y2) * cross) / (6.0 * area)
            centroid = np.array([cx, cy])
            
            pts[i] = pts[i] + step_size * (centroid - pts[i])
            moved = True
        
        if not moved:
            break
    
    return pts

def _regular_polygon_hull(n_hull, center, radius):
    """Generate a regular n-gon centered at 'center' with given radius."""
    angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _place_interior_points_on_circle(n_interior, center, radius):
    """Place interior points on a smaller concentric circle."""
    if n_interior == 0:
        return np.empty((0, 2))
    angles = np.linspace(0, 2 * np.pi, n_interior, endpoint=False) + np.pi / n_interior
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _build_initial_config(n_hull=7, n_interior=6):
    """
    Build initial configuration: regular hull polygon + interior points on concentric circle.
    """
    center = np.array([0.5, 0.5])
    hull_radius = 0.45
    interior_radius = 0.25
    
    hull_pts = _regular_polygon_hull(n_hull, center, hull_radius)
    interior_pts = _place_interior_points_on_circle(n_interior, center, interior_radius)
    
    return np.vstack([hull_pts, interior_pts])

def _optimize_all_points(points0, max_iter=2000, alpha=100.0):
    """Optimize all 13 points using L-BFGS-B with soft-min objective."""
    def objective(pts_flat):
        return _soft_score_objective(pts_flat, alpha)
    
    bounds = [(0.01, 0.99)] * (2 * len(points0))
    
    try:
        result = minimize(
            objective,
            points0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points0

def _nelder_mead_optimize(points0, max_iter=3000):
    """Nelder-Mead optimization."""
    def objective(pts_flat):
        return _soft_score_objective(pts_flat, alpha=80.0)
    
    try:
        result = minimize(
            objective,
            points0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points0

def _coordinate_descent_polish(points0, max_sweeps=60):
    """Fine-grained coordinate descent polish."""
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001]
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 15:
            break
        
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local = best_score
            
            for step in step_sizes:
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step*0.707, step*0.707), (-step*0.707, step*0.707),
                               (step*0.707, -step*0.707), (-step*0.707, -step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local + 1e-12:
                        best_local = s
                        best_x, best_y = nx, ny
                        improved = True
            
            pts[i] = [best_x, best_y]
        
        cur = _score(pts)
        if cur > best_score + 1e-12:
            best_score = cur
            best_pts = pts.copy()
        
        if not improved:
            break
    
    return best_pts, best_score

def _try_configurations():
    """Try multiple hull/interior configurations and return the best."""
    best_points = None
    best_score = -np.inf
    
    configs = [
        (7, 6),  # 7 hull + 6 interior
        (8, 5),  # 8 hull + 5 interior
        (6, 7),  # 6 hull + 7 interior
        (9, 4),  # 9 hull + 4 interior
        (5, 8),  # 5 hull + 8 interior
    ]
    
    for n_hull, n_interior in configs:
        if time.time() > _DEADLINE - 120:
            break
        
        # Build initial config
        pts = _build_initial_config(n_hull, n_interior)
        
        # Lloyd relaxation on interior points
        if time.time() < _DEADLINE - 90:
            lloyd_pts = _lloyd_relax_interior(pts, n_iter=40, step_size=0.4)
            s = _score(lloyd_pts)
            if s > best_score:
                best_score = s
                best_points = lloyd_pts.copy()
        
        # Optimize all points
        if best_points is not None and time.time() < _DEADLINE - 60:
            opt_pts = _optimize_all_points(best_points, max_iter=1500, alpha=100.0)
            s = _score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        
        # Nelder-Mead refinement
        if best_points is not None and time.time() < _DEADLINE - 45:
            nm_pts = _nelder_mead_optimize(best_points, max_iter=2000)
            s = _score(nm_pts)
            if s > best_score:
                best_score = s
                best_points = nm_pts.copy()
    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    best_points = None
    best_score = -np.inf
    
    # Try multiple configurations
    if time.time() < _DEADLINE - 150:
        cfg_pts, cfg_score = _try_configurations()
        if cfg_pts is not None and cfg_score > best_score:
            best_score = cfg_score
            best_points = cfg_pts.copy()
    
    # Additional optimization passes on the best found
    if best_points is not None and time.time() < _DEADLINE - 60:
        # Another L-BFGS pass with different alpha
        opt_pts = _optimize_all_points(best_points, max_iter=1000, alpha=150.0)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
    
    if best_points is not None and time.time() < _DEADLINE - 40:
        # Nelder-Mead
        nm_pts = _nelder_mead_optimize(best_points, max_iter=2000)
        s = _score(nm_pts)
        if s > best_score:
            best_score = s
            best_points = nm_pts.copy()
    
    # Final coordinate descent polish
    if best_points is not None and time.time() < _DEADLINE - 20:
        polished_pts, polished_s = _coordinate_descent_polish(best_points, max_sweeps=50)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_initial_config(7, 6)
    
    return best_points

# EVOLVE-BLOCK-END
