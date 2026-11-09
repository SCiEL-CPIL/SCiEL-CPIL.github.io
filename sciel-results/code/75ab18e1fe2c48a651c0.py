# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 590.0
...

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
    """Smooth surrogate: -soft_min(triangle_areas) / hull_area.
    soft_min via log-sum-exp: -1/beta * log(sum(exp(-beta * a_i)))
    This is differentiable and approximates min(a_i) as beta -> inf.
    """
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
        if np.any(areas <= 0):
            return 1e6
        # Soft min: -1/beta * log(sum(exp(-beta * areas)))
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min = -log_sum_exp / beta
        return -soft_min / ha
    except Exception:
        return 1e6

def _epigraph_objective(x, beta=100.0):
    """Epigraph formulation: minimize -t + penalty for t > a_i.
    x = [points_flat..., t]
    Penalty: sum(max(0, t - a_i)^2) / (n_triangles)
    This makes the problem smooth and handles the min constraint naturally.
    """
    n_pts = 13
    pts = x[:2*n_pts].reshape(-1, 2)
    t = x[2*n_pts]
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
        if np.any(areas <= 0):
            return 1e6
        # Penalty for t exceeding any triangle area
        violations = np.maximum(0.0, t - areas)
        penalty = np.sum(violations ** 2) / len(areas)
        # We want to maximize t/hull_area, so minimize -t/ha + penalty
        return -t / ha + beta * penalty
    except Exception:
        return 1e6

def _optimize_epigraph(points0, betas=(50, 100, 200, 400), max_iter_per_beta=300):
    """Optimize using epigraph formulation with progressive beta."""
    n = len(points0)
    # Initial t estimate
    t0 = _min_triangle_area(points0) * 0.9
    x0 = np.concatenate([points0.flatten(), [t0]])
    bounds = [(0.05, 0.95)] * (2 * n) + [(0.0, 0.5)]
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        try:
            result = minimize(
                _epigraph_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'maxfun': max_iter_per_beta * 4}
            )
            if result.fun < _epigraph_objective(x0, beta):
                x0 = result.x
        except Exception:
            pass
    
    return x0[:2*n].reshape(-1, 2)

def _optimize_soft_min(points0, betas=(50, 100, 200, 400, 600), max_iter_per_beta=200):
    """Optimize using progressive beta annealing with L-BFGS-B."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        try:
            result = minimize(
                _soft_min_objective,
                pts,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'maxfun': max_iter_per_beta * 3}
            )
            if result.fun < _soft_min_objective(pts, beta):
                pts = result.x
        except Exception:
            pass
    
    return pts.reshape(-1, 2)

def _polish(points0, max_iter=500):
    """Final Powell polish with tight tolerances."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'xtol': 1e-14, 'ftol': 1e-16, 'maxfev': max_iter * 10}
        )
        if result.fun < _negative_ratio(pts):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _targeted_worst_triangle_polish(points0, max_rounds=50):
    """Identify the worst (smallest area) triangles and do targeted local moves
    on their vertices to increase their area, while monitoring overall score."""
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE - 15:
            break
        
        # Find worst triangles
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Get the 5 worst triangles
        worst_idx = np.argsort(areas)[:5]
        worst_triangles = idx[worst_idx]
        
        improved = False
        for tri in worst_triangles:
            for vi in range(3):
                pi = tri[vi]
                x0, y0 = pts[pi]
                # Try small moves in multiple directions
                for step in [0.005, 0.002, 0.001]:
                    for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                        dx = step * np.cos(angle)
                        dy = step * np.sin(angle)
                        nx, ny = x0 + dx, y0 + dy
                        if nx < 0.02 or nx > 0.98 or ny < 0.02 or ny > 0.98:
                            continue
                        new_pts = pts.copy()
                        new_pts[pi] = [nx, ny]
                        s = _score(new_pts)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_pts = new_pts.copy()
                            pts = new_pts.copy()
                            x0, y0 = nx, ny
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return best_pts

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

def _lloyd_relax(points, max_iters=50):
    """Lloyd's algorithm: relax points to Voronoi centroids.
    This enforces more uniform spacing, which helps maximize min triangle area."""
    pts = points.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 30:
            break
        try:
            vor = Voronoi(pts)
            new_pts = pts.copy()
            for i in range(n):
                region = vor.regions[vor.point_region[i]]
                if -1 in region:
                    # Unbounded region; keep original point
                    continue
                vertices = vor.vertices[region]
                # Clip to [0.05, 0.95]
                vertices = np.clip(vertices, 0.05, 0.95)
                centroid = np.mean(vertices, axis=0)
                # Only move if centroid is finite and within bounds
                if np.all(np.isfinite(centroid)):
                    new_pts[i] = centroid
            # Check if converged
            if np.max(np.abs(new_pts - pts)) < 1e-10:
                break
            pts = new_pts
        except Exception:
            break
    
    return pts

def _lloyd_then_optimize(points0):
    """First Lloyd-relax, then optimize with soft-min."""
    relaxed = _lloyd_relax(points0, max_iters=30)
    return _optimize_soft_min(relaxed, betas=(50, 100, 200, 400, 600), max_iter_per_beta=150)

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
    
    # 8. Hexagonal packing-inspired: 7 in hex pattern + 6 more
    hex_pts = np.array([
        [0.5, 0.5],
        [0.5+0.2, 0.5], [0.5-0.2, 0.5],
        [0.5+0.1, 0.5+0.173], [0.5-0.1, 0.5+0.173],
        [0.5+0.1, 0.5-0.173], [0.5-0.1, 0.5-0.173],
        [0.5+0.35, 0.5+0.2], [0.5-0.35, 0.5+0.2],
        [0.5+0.35, 0.5-0.2], [0.5-0.35, 0.5-0.2],
        [0.5, 0.5+0.38], [0.5, 0.5-0.38]
    ])
    starts.append(hex_pts)
    
    # 9. Two concentric rings: 5 inner, 8 outer
    inner5 = _regular_polygon(5, radius=0.22, center=(0.5, 0.5))
    outer8 = _regular_polygon(8, radius=0.46, center=(0.5, 0.5))
    starts.append(np.vstack([inner5, outer8]))
    
    # 10. Three concentric rings: 3 inner, 4 mid, 6 outer
    inner3r = _regular_polygon(3, radius=0.15, center=(0.5, 0.5))
    mid4r = _regular_polygon(4, radius=0.32, center=(0.5, 0.5))
    outer6r = _regular_polygon(6, radius=0.47, center=(0.5, 0.5))
    starts.append(np.vstack([inner3r, mid4r, outer6r]))
    
    # For each start, try multiple optimization strategies
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        
        # Strategy A: Soft-min with progressive beta annealing
        if time.time() < _DEADLINE - 60:
            sm_pts = _optimize_soft_min(start, betas=(50, 100, 200, 400, 600), max_iter_per_beta=150)
            _update(sm_pts)
        
        # Strategy B: Powell polish on soft-min result
        if time.time() < _DEADLINE - 40:
            polish_pts = _polish(sm_pts, max_iter=300)
            _update(polish_pts)
        
        # Strategy C: Epigraph optimization
        if time.time() < _DEADLINE - 60:
            epi_pts = _optimize_epigraph(start, betas=(50, 100, 200, 400), max_iter_per_beta=200)
            _update(epi_pts)
        
        # Strategy D: Powell polish on epigraph result
        if time.time() < _DEADLINE - 40:
            polish_epi = _polish(epi_pts, max_iter=200)
            _update(polish_epi)
        
        # Strategy E: Coordinate descent
        if time.time() < _DEADLINE - 30:
            cd_pts, cd_score = _coordinate_descent(start, max_sweeps=80)
            _update(cd_pts)
        
        # Strategy F: Soft-min on coordinate descent result
        if time.time() < _DEADLINE - 40:
            sm_pts2 = _optimize_soft_min(cd_pts, betas=(100, 200, 400), max_iter_per_beta=100)
            _update(sm_pts2)
        
        # Strategy G: Final Powell polish
        if time.time() < _DEADLINE - 30:
            polish_pts2 = _polish(sm_pts2, max_iter=200)
            _update(polish_pts2)
        
        # Strategy H: Lloyd relaxation then soft-min
        if time.time() < _DEADLINE - 50:
            lloyd_pts = _lloyd_then_optimize(start)
            _update(lloyd_pts)
        
        # Strategy I: Lloyd then coordinate descent
        if time.time() < _DEADLINE - 30:
            lloyd_cd_pts, _ = _coordinate_descent(lloyd_pts, max_sweeps=60)
            _update(lloyd_cd_pts)
    
    # Final targeted polish on best result
    if best_points is not None and time.time() < _DEADLINE - 20:
        targeted_pts = _targeted_worst_triangle_polish(best_points, max_rounds=50)
        _update(targeted_pts)
    
    # Final high-precision Powell polish
    if best_points is not None and time.time() < _DEADLINE - 15:
        final_pts = _polish(best_points, max_iter=400)
        _update(final_pts)
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
