# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
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

def _negative_ratio(points_flat):
    """Objective to minimize: -min_area/hull_area"""
    pts = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        ma = _min_triangle_area(pts)
        if ma <= 0:
            return 1e6
        return -ma / ha
    except Exception:
        return 1e6

def _soft_min_objective(points_flat, beta=200.0):
    pts = points_flat.reshape(-1, 2)
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
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        soft_min = -1.0 / beta * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
        if soft_min <= 0:
            return 1e6
        return -soft_min / ha
    except Exception:
        return 1e6

def _voronoi_centroid(points, i):
    """Compute Voronoi cell centroid for point i."""
    try:
        vor = Voronoi(points)
        region_idx = i
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) == 0:
            # Unbounded region, fallback to centroid of neighbors via Delaunay
            tri = Delaunay(points)
            mask = np.any(tri.simplices == i, axis=1)
            if not np.any(mask):
                return points[i].copy()
            neighbors = np.unique(tri.simplices[mask].reshape(-1))
            neighbors = neighbors[neighbors != i]
            if len(neighbors) > 0:
                return np.mean(points[neighbors], axis=0)
            return points[i].copy()
        
        vertices = vor.vertices[region]
        
        if len(vertices) < 3:
            return points[i].copy()
        
        # Compute centroid of the polygon
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

def _hull_indices(points):
    """Return set of indices of points on the convex hull."""
    try:
        hull = ConvexHull(points)
        return set(hull.vertices)
    except Exception:
        return set(range(len(points)))

def _lloyd_iteration(points, n_iter=30, step_size=0.5):
    """Lloyd's algorithm: move interior points to their Voronoi cell centroids."""
    pts = points.copy().astype(float)
    n = len(pts)
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        hull_idx = _hull_indices(pts)
        
        for i in range(n):
            if i in hull_idx:
                continue
            centroid = _voronoi_centroid(pts, i)
            new_pos = pts[i] + step_size * (centroid - pts[i])
            new_pos = np.clip(new_pos, 0.02, 0.98)
            pts[i] = new_pos
        
        # Check if we need to break due to invalidity
        if _score(pts) <= 0:
            pts = pts.copy()
            # Try to recover by moving points slightly
            for i in range(n):
                if i in _hull_indices(pts):
                    continue
                pts[i] = pts[i] + np.array([0.001, 0.001])
                pts[i] = np.clip(pts[i], 0.02, 0.98)
            break
            
    return pts

def _optimize_scipy(points0, max_iter=500, method='L-BFGS-B'):
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.02, 0.98)] * (2 * n)
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 2, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.success or result.fun < _negative_ratio(pts):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _optimize_soft_min(points0, beta=200.0, max_iter=400):
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.02, 0.98)] * (2 * n)
    try:
        result = minimize(
            _soft_min_objective,
            pts,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.success or result.fun < _soft_min_objective(pts, beta):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _polish(points0, max_sweeps=200, step_sizes=None):
    """High-precision coordinate descent polish."""
    if step_sizes is None:
        step_sizes = [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 15:
            break
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local_score = best_score
            
            for step in step_sizes:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-13:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-13:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _coordinate_descent(points0, max_sweeps=150, step_sizes=None):
    """Cyclic coordinate descent."""
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 25:
            break
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local_score = best_score
            
            for step in step_sizes:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.02 or nx > 0.98 or ny < 0.02 or ny > 0.98:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-11:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-11:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _generate_starts(n=13, rng=None):
    """Generate diverse starting configurations."""
    if rng is None:
        rng = np.random.default_rng(seed=42)
    
    starts = []
    
    # Regular 13-gon
    starts.append(_regular_polygon(13, radius=0.48, center=(0.5, 0.5)))
    starts.append(_regular_polygon(13, radius=0.5, center=(0.5, 0.5)))
    
    # Ring + interior configurations
    for n_outer in range(8, 13):
        n_inner = n - n_outer
        outer = _regular_polygon(n_outer, radius=0.48, center=(0.5, 0.5))
        if n_inner > 0:
            angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
            inner_r = 0.48 / (n_outer * 0.35)
            inner = np.column_stack([0.5 + inner_r*np.cos(angles), 0.5 + inner_r*np.sin(angles)])
            starts.append(np.vstack([outer, inner]))
        else:
            starts.append(outer)
    
    # Random starts
    for _ in range(8):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
    
    # Grid-like start
    grid_x = np.linspace(0.15, 0.85, 4)
    grid_y = np.linspace(0.15, 0.85, 4)
    grid_pts = np.array([[x, y] for x in grid_x for y in grid_y])[:n]
    starts.append(grid_pts)
    
    # Two concentric rings
    outer_ring = _regular_polygon(7, radius=0.48, center=(0.5, 0.5))
    inner_ring = _regular_polygon(6, radius=0.25, center=(0.5, 0.5))
    starts.append(np.vstack([outer_ring, inner_ring]))
    
    # Triangle-based start: 3 outer corners + 10 distributed
    tri_outer = np.array([[0.5, 0.98], [0.1, 0.15], [0.9, 0.15]])
    remaining = n - 3
    angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
    inner = np.column_stack([0.5 + 0.3*np.cos(angles + 0.5), 0.5 + 0.3*np.sin(angles + 0.5)])
    starts.append(np.vstack([tri_outer, inner]))
    
    return starts

def _refine(points, label=""):
    """Apply a sequence of refinement steps to a point set."""
    best = points.copy()
    best_s = _score(points)
    
    def _try(pts):
        nonlocal best, best_s
        s = _score(pts)
        if s > best_s:
            best_s = s
            best = pts.copy()
        return best
    
    # Step 1: Lloyd's algorithm with Voronoi centroids
    if time.time() < _DEADLINE - 60:
        lloyd_pts = _lloyd_iteration(best, n_iter=20, step_size=0.5)
        _try(lloyd_pts)
    
    # Step 2: Scipy optimization
    if time.time() < _DEADLINE - 60:
        opt_pts = _optimize_scipy(best, max_iter=300)
        _try(opt_pts)
    
    # Step 3: Soft-min optimization
    if time.time() < _DEADLINE - 50:
        soft_pts = _optimize_soft_min(best, beta=300, max_iter=300)
        _try(soft_pts)
    
    # Step 4: Another Lloyd pass
    if time.time() < _DEADLINE - 40:
        lloyd_pts2 = _lloyd_iteration(best, n_iter=20, step_size=0.5)
        _try(lloyd_pts2)
    
    # Step 5: Coordinate descent
    if time.time() < _DEADLINE - 30:
        cd_pts, _ = _coordinate_descent(best, max_sweeps=80)
        _try(cd_pts)
    
    # Step 6: High-precision polish
    if time.time() < _DEADLINE - 15:
        pol_pts, _ = _polish(best, max_sweeps=150)
        _try(pol_pts)
    
    return best, best_s

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    def _update(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    starts = _generate_starts(n, rng)
    
    for si, start in enumerate(starts):
        if time.time() > _DEADLINE - 30:
            break
        
        # First do a quick coordinate descent to get a better starting point
        cd_pts, _ = _coordinate_descent(start, max_sweeps=30)
        _update(cd_pts)
        
        # Then full refinement
        refined, refined_s = _refine(cd_pts, label=f"start{si}")
        _update(refined)
    
    # Final polish on the overall best
    if best_points is not None and time.time() < _DEADLINE - 10:
        final_pts, _ = _polish(best_points, max_sweeps=200)
        _update(final_pts)
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
