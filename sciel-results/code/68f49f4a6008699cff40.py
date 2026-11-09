# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import differential_evolution, minimize
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

def _delaunay_centroid(points, i):
    """Compute Voronoi cell centroid for point i using Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        # Find triangles containing point i
        mask = np.any(tri.simplices == i, axis=1)
        if not np.any(mask):
            return points[i].copy()
        
        # Get neighboring points
        neighbors = np.unique(tri.simplices[mask].reshape(-1))
        neighbors = neighbors[neighbors != i]
        
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Find the Voronoi region for point i
        region_idx = i
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) == 0:
            # Unbounded region, fallback to centroid of neighbors
            if len(neighbors) > 0:
                return np.mean(points[neighbors], axis=0)
            return points[i].copy()
        
        # Get vertices of the Voronoi cell
        vertices = vor.vertices[region]
        
        if len(vertices) < 3:
            return points[i].copy()
        
        # Compute centroid of the polygon
        # Use the formula for polygon centroid
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        # Shoelace formula for area and centroid
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

def _lloyd_iteration(points, n_iter=50, step_size=0.5):
    """Lloyd's algorithm: move each interior point toward its Voronoi centroid."""
    pts = points.copy()
    n = len(pts)
    
    # Identify hull points
    try:
        hull = ConvexHull(pts)
        hull_idx = set(hull.vertices)
    except Exception:
        return pts
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        for i in range(n):
            if i in hull_idx:
                continue
            
            centroid = _delaunay_centroid(pts, i)
            # Move toward centroid with damping
            new_pos = pts[i] + step_size * (centroid - pts[i])
            
            # Keep within bounds
            new_pos = np.clip(new_pos, 0.01, 0.99)
            pts[i] = new_pos
        
        # Recompute hull after each iteration
        try:
            hull = ConvexHull(pts)
            hull_idx = set(hull.vertices)
        except Exception:
            break
    
    return pts

def _optimize_soft_min(points0, max_iter=200):
    """Optimize using a soft-min objective for smoother gradients."""
    def soft_min_objective(pts_flat, alpha=100):
        pts = pts_flat.reshape(-1, 2)
        try:
            ha = _hull_area(pts)
            if ha <= 0:
                return 1e6
            # Compute all triangle areas
            n = len(pts)
            idx = np.array(list(itertools.combinations(range(n), 3)))
            a = pts[idx[:, 0]]
            b = pts[idx[:, 1]]
            c = pts[idx[:, 2]]
            areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                                 + b[:, 0] * (c[:, 1] - a[:, 1])
                                 + c[:, 0] * (a[:, 1] - b[:, 1]))
            # Soft minimum
            log_sum_exp = np.log(np.sum(np.exp(-alpha * areas))) / alpha
            soft_min_area = -log_sum_exp
            return -(soft_min_area / ha)
        except Exception:
            return 1e6
    
    try:
        result = minimize(
            soft_min_objective,
            points0.flatten(),
            method='L-BFGS-B',
            bounds=[(0.01, 0.99)] * (2 * len(points0)),
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points0

def _coordinate_descent_polish(points0, max_sweeps=100):
    """Coordinate descent with adaptive step sizes."""
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
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

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Start with regular polygon, then Lloyd + soft-min + polish
    start = _regular_polygon(n, radius=0.45, center=(0.5, 0.5))
    
    # Lloyd iteration to balance interior points
    lloyd_pts = _lloyd_iteration(start, n_iter=30, step_size=0.3)
    s = _score(lloyd_pts)
    if s > best_score:
        best_score = s
        best_points = lloyd_pts.copy()
    
    # Soft-min optimization
    if time.time() < _DEADLINE - 60:
        soft_pts = _optimize_soft_min(lloyd_pts, max_iter=100)
        s = _score(soft_pts)
        if s > best_score:
            best_score = s
            best_points = soft_pts.copy()
    
    # Coordinate descent polish
    if best_points is not None and time.time() < _DEADLINE - 30:
        polished_pts, polished_s = _coordinate_descent_polish(best_points, max_sweeps=200)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Strategy 2: Differential Evolution as backup
    if best_score < 0.15 and time.time() < _DEADLINE - 120:
        def de_obj(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            try:
                sc = _score(pts)
                return -sc if sc > 0 else 1.0
            except Exception:
                return 1.0
        
        bounds = [(0.01, 0.99)] * (2 * n)
        
        def time_cb(xk, fun, conv):
            return time.time() > _DEADLINE - 60
        
        try:
            result = differential_evolution(
                de_obj, bounds, seed=42, maxiter=500, popsize=15,
                tol=1e-4, callback=time_cb, polish=False
            )
            if result.x is not None:
                de_pts = result.x.reshape(-1, 2)
                s = _score(de_pts)
                if s > best_score:
                    best_score = s
                    best_points = de_pts.copy()
        except Exception:
            pass
    
    # Final polish
    if best_points is not None and time.time() < _DEADLINE - 15:
        final_pts, final_s = _coordinate_descent_polish(best_points, max_sweeps=300)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
