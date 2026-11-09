# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 590.0

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

def _soft_min_objective(points_flat, beta=200.0):
    """Smooth surrogate: -soft_min(triangle_areas) / hull_area."""
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

def _lloyd_iteration(points, n_iter=50, hull_margin=0.02):
    """Lloyd's algorithm: move interior points to Voronoi cell centroids.
    Only move points that are NOT on the convex hull."""
    pts = points.copy().astype(float)
    n = len(pts)
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        try:
            hull = ConvexHull(pts)
            hull_idx = set(hull.vertices)
        except Exception:
            break
        
        for i in range(n):
            if i in hull_idx:
                continue
            try:
                tri = Delaunay(pts)
                tri_idx = tri.simplices
                masks = []
                for t in tri_idx:
                    if i in t:
                        masks.append(t)
                if len(masks) == 0:
                    continue
                voronoi_pts = []
                for t in masks:
                    p1, p2, p3 = pts[t[0]], pts[t[1]], pts[t[2]]
                    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if abs(d) < 1e-12:
                        continue
                    ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + 
                          (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + 
                          (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
                    uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + 
                          (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + 
                          (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
                    voronoi_pts.append([ux, uy])
                if len(voronoi_pts) >= 3:
                    voronoi_pts = np.array(voronoi_pts)
                    centroid = np.mean(voronoi_pts, axis=0)
                    centroid[0] = np.clip(centroid[0], hull_margin, 1 - hull_margin)
                    centroid[1] = np.clip(centroid[1], hull_margin, 1 - hull_margin)
                    pts[i] = 0.5 * pts[i] + 0.5 * centroid
            except Exception:
                continue
    return pts

def _optimize_with_scipy(points0, max_iter=500):
    """Use scipy's L-BFGS-B to optimize all 13 points simultaneously."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 2}
        )
        if result.success or result.fun < _negative_ratio(pts):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _optimize_soft_min(points0, beta=200.0, max_iter=300):
    """Optimize using soft-min surrogate with L-BFGS-B."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    try:
        result = minimize(
            _soft_min_objective,
            pts,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < _soft_min_objective(pts, beta):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _polish(points0, max_sweeps=500, step_sizes=None):
    """High-precision coordinate descent polish."""
    if step_sizes is None:
        step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
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
                    if nx < 0.02 or nx > 0.98 or ny < 0.02 or ny > 0.98:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-12:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _coordinate_descent(points0, max_sweeps=200, step_sizes=None):
    """Cyclic coordinate descent on each point's x and y."""
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
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
                    if nx < 0.05 or nx > 0.95 or ny < 0.05 or ny > 0.95:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-10:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-10:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _jitter_and_reoptimize(points, rng, jitter_scale=0.05, n_restarts=3):
    """Perturb points and re-optimize to escape local optima."""
    best_pts = points.copy()
    best_score = _score(points)
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 40:
            break
        perturbed = points + rng.uniform(-jitter_scale, jitter_scale, points.shape)
        perturbed = np.clip(perturbed, 0.05, 0.95)
        opt_pts = _optimize_soft_min(perturbed, beta=150.0, max_iter=150)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_pts = opt_pts.copy()
    return best_pts, best_score

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
    
    # Multiple starting configurations
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(13, radius=0.48, center=(0.5, 0.5)))
    
    # 2. Ring + interior: 9 outer, 4 inner
    outer9 = _regular_polygon(9, radius=0.48, center=(0.5, 0.5))
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner4 = np.column_stack([0.5 + 0.2*np.cos(angles4), 0.5 + 0.2*np.sin(angles4)])
    starts.append(np.vstack([outer9, inner4]))
    
    # 3. Ring + interior: 10 outer, 3 inner
    outer10 = _regular_polygon(10, radius=0.48, center=(0.5, 0.5))
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner3 = np.column_stack([0.5 + 0.18*np.cos(angles3), 0.5 + 0.18*np.sin(angles3)])
    starts.append(np.vstack([outer10, inner3]))
    
    # 4. Ring + interior: 11 outer, 2 inner
    outer11 = _regular_polygon(11, radius=0.48, center=(0.5, 0.5))
    inner2 = np.array([[0.5 + 0.15, 0.5], [0.5 - 0.15, 0.5]])
    starts.append(np.vstack([outer11, inner2]))
    
    # 5. Ring + interior: 12 outer, 1 inner
    outer12 = _regular_polygon(12, radius=0.48, center=(0.5, 0.5))
    inner1 = np.array([[0.5, 0.5]])
    starts.append(np.vstack([outer12, inner1]))
    
    # 6. All 13 on hull (regular)
    starts.append(_regular_polygon(13, radius=0.5, center=(0.5, 0.5)))
    
    # 7. Random starts
    for _ in range(5):
        starts.append(rng.uniform(0.1, 0.9, (13, 2)))
    
    # 8. Lloyd-initialized: start with random, apply Lloyd, then optimize
    for _ in range(3):
        random_pts = rng.uniform(0.15, 0.85, (13, 2))
        lloyd_pts = _lloyd_iteration(random_pts, n_iter=30)
        starts.append(lloyd_pts)
    
    # For each start, try multiple optimization strategies
    for si, start in enumerate(starts):
        if time.time() > _DEADLINE - 40:
            break
        
        # Strategy A: Scipy L-BFGS-B on exact ratio
        if time.time() < _DEADLINE - 80:
            opt_pts = _optimize_with_scipy(start, max_iter=250)
            _update(opt_pts)
        
        # Strategy B: Soft-min optimization (smooth surrogate)
        if time.time() < _DEADLINE - 60:
            soft_pts = _optimize_soft_min(start, beta=200.0, max_iter=300)
            _update(soft_pts)
        
        # Strategy C: Coordinate descent
        if time.time() < _DEADLINE - 40:
            cd_pts, cd_score = _coordinate_descent(soft_pts, max_sweeps=80)
            _update(cd_pts)
        
        # Strategy D: Soft-min on coordinate descent result
        if time.time() < _DEADLINE - 35:
            soft_pts2 = _optimize_soft_min(cd_pts, beta=300.0, max_iter=200)
            _update(soft_pts2)
        
        # Strategy E: Jitter and re-optimize
        if time.time() < _DEADLINE - 30:
            jitter_pts, jitter_score = _jitter_and_reoptimize(soft_pts2, rng, jitter_scale=0.03, n_restarts=2)
            _update(jitter_pts)
    
    # Final global polish on the best result found
    if best_points is not None and time.time() < _DEADLINE - 15:
        final_pts, final_score = _polish(best_points, max_sweeps=250)
        _update(final_pts)
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
