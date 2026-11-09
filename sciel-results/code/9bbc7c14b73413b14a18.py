# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize, differential_evolution
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

def _voronoi_centroid(points, i):
    """Compute Voronoi cell centroid for point i."""
    try:
        vor = Voronoi(points)
        region_idx = i
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) == 0:
            # Unbounded region - use Delaunay neighbors
            try:
                tri = Delaunay(points)
                mask = np.any(tri.simplices == i, axis=1)
                if np.any(mask):
                    neighbors = np.unique(tri.simplices[mask].reshape(-1))
                    neighbors = neighbors[neighbors != i]
                    if len(neighbors) > 0:
                        return np.mean(points[neighbors], axis=0)
            except Exception:
                pass
            return points[i].copy()
        
        vertices = vor.vertices[region]
        if len(vertices) < 3:
            return points[i].copy()
        
        # Polygon centroid via shoelace
        x = vertices[:, 0]
        y = vertices[:, 1]
        x2 = np.roll(x, -1)
        y2 = np.roll(y, -1)
        cross = x * y2 - x2 * y
        area = 0.5 * np.sum(cross)
        
        if abs(area) < 1e-12:
            return np.mean(vertices, axis=0)
        
        cx = np.sum((x + x2) * cross) / (6.0 * area)
        cy = np.sum((y + y2) * cross) / (6.0 * area)
        return np.array([cx, cy])
    except Exception:
        return points[i].copy()

def _lloyd_interior_only(points, n_iter=40, step_size=0.5):
    """Lloyd's algorithm: only move interior points toward Voronoi centroids."""
    pts = points.copy()
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
        hull_idx = set(hull.vertices)
    except Exception:
        return pts
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        # Recompute Voronoi once per iteration
        try:
            vor = Voronoi(pts)
        except Exception:
            break
        
        for i in range(n):
            if i in hull_idx:
                continue
            
            region = vor.regions[i]
            if -1 in region or len(region) == 0:
                # Fallback to Delaunay neighbors
                try:
                    tri = Delaunay(pts)
                    mask = np.any(tri.simplices == i, axis=1)
                    if np.any(mask):
                        neighbors = np.unique(tri.simplices[mask].reshape(-1))
                        neighbors = neighbors[neighbors != i]
                        if len(neighbors) > 0:
                            centroid = np.mean(pts[neighbors], axis=0)
                            pts[i] = pts[i] + step_size * (centroid - pts[i])
                            pts[i] = np.clip(pts[i], 0.01, 0.99)
                except Exception:
                    pass
                continue
            
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                continue
            
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
            pts[i] = np.clip(pts[i], 0.01, 0.99)
        
        # Recompute hull
        try:
            hull = ConvexHull(pts)
            hull_idx = set(hull.vertices)
        except Exception:
            break
    
    return pts

def _nelder_mead_optimize(points0, max_iter=5000):
    """Nelder-Mead optimization of A_min/A_hull ratio."""
    def objective(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        try:
            ha = _hull_area(pts)
            if ha <= 0:
                return 1e6
            ma = _min_triangle_area(pts)
            if ma <= 0:
                return 1e6
            return -(ma / ha)
        except Exception:
            return 1e6
    
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

def _lbfgs_optimize(points0, max_iter=200):
    """L-BFGS-B optimization with soft-min objective for smoother gradients."""
    def soft_min_objective(pts_flat, alpha=80):
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
            # Soft minimum
            max_neg = np.max(-alpha * areas)
            log_sum_exp = np.log(np.sum(np.exp(-alpha * areas - max_neg))) + max_neg
            soft_min_area = -log_sum_exp / alpha
            return -(soft_min_area / ha)
        except Exception:
            return 1e6
    
    try:
        result = minimize(
            soft_min_objective,
            points0.flatten(),
            method='L-BFGS-B',
            bounds=[(0.01, 0.99)] * (2 * len(points0)),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points0

def _coordinate_descent_polish(points0, max_sweeps=50):
    """Fine-grained coordinate descent polish."""
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    step_sizes = [0.002, 0.001, 0.0005, 0.0002]
    
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

def _good_starting_config():
    """Regular 12-gon on hull + 1 center point."""
    n_hull = 12
    angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    radius = 0.45
    center = (0.5, 0.5)
    hull_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)])
    # Add center
    pts = np.vstack([hull_pts, np.array([center])])
    return pts

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Start with regular 12-gon + center
    start1 = _good_starting_config()
    s = _score(start1)
    if s > best_score:
        best_score = s
        best_points = start1.copy()
    
    # Lloyd relaxation on interior points
    if time.time() < _DEADLINE - 60:
        lloyd_pts = _lloyd_interior_only(start1, n_iter=30, step_size=0.4)
        s = _score(lloyd_pts)
        if s > best_score:
            best_score = s
            best_points = lloyd_pts.copy()
    
    # Nelder-Mead optimization (most effective per cell findings)
    if best_points is not None and time.time() < _DEADLINE - 90:
        nm_pts = _nelder_mead_optimize(best_points, max_iter=3000)
        s = _score(nm_pts)
        if s > best_score:
            best_score = s
            best_points = nm_pts.copy()
    
    # L-BFGS with soft-min
    if best_points is not None and time.time() < _DEADLINE - 60:
        lbfgs_pts = _lbfgs_optimize(best_points, max_iter=150)
        s = _score(lbfgs_pts)
        if s > best_score:
            best_score = s
            best_points = lbfgs_pts.copy()
    
    # Another Nelder-Mead pass
    if best_points is not None and time.time() < _DEADLINE - 45:
        nm2_pts = _nelder_mead_optimize(best_points, max_iter=2000)
        s = _score(nm2_pts)
        if s > best_score:
            best_score = s
            best_points = nm2_pts.copy()
    
    # Coordinate descent polish
    if best_points is not None and time.time() < _DEADLINE - 20:
        polished_pts, polished_s = _coordinate_descent_polish(best_points, max_sweeps=40)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Strategy 2: Try a different starting configuration
    if time.time() < _DEADLINE - 120:
        # Regular 13-gon
        angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        radius13 = 0.45
        center13 = (0.5, 0.5)
        start2 = np.column_stack([center13[0] + radius13 * np.cos(angles13),
                                  center13[1] + radius13 * np.sin(angles13)])
        s = _score(start2)
        if s > best_score:
            best_score = s
            best_points = start2.copy()
        
        if time.time() < _DEADLINE - 60:
            nm3_pts = _nelder_mead_optimize(start2, max_iter=2000)
            s = _score(nm3_pts)
            if s > best_score:
                best_score = s
                best_points = nm3_pts.copy()
    
    # Final polish
    if best_points is not None and time.time() < _DEADLINE - 10:
        final_pts, final_s = _coordinate_descent_polish(best_points, max_sweeps=30)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _good_starting_config()
    
    return best_points

# EVOLVE-BLOCK-END
