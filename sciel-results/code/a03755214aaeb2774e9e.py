import time
import numpy as np
from scipy.optimize import minimize

N = 13

def _convex_hull_area(pts):
    """Compute area of convex hull of pts (n,2)."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: if all points collinear or degenerate
        return 1e-10

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(pts):
    """Score = min_triangle_area / convex_hull_area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_objective(flat_pts, tau):
    """
    Objective to minimize: -soft_min(triangle_areas) / hull_area
    Using log-sum-exp for soft min of areas.
    """
    pts = flat_pts.reshape(-1, 2)
    n = len(pts)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    
    # Soft min: -log(sum(exp(-area/tau))) * tau
    # This approximates min(areas) as tau -> 0
    max_neg = np.max(-areas / tau)
    soft_min = -tau * (max_neg + np.log(np.sum(np.exp(-areas / tau - max_neg))))
    
    # Hull area
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    # We want to maximize soft_min / hull_area, so minimize negative
    return -(soft_min / hull_area)

def _soft_min_objective_fast(flat_pts, tau):
    """Faster vectorized version of soft min objective."""
    pts = flat_pts.reshape(-1, 2)
    n = len(pts)
    
    # Compute all triangle areas vectorized
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # Use broadcasting
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    i_idx = i_idx[i_idx < j_idx]
    j_idx = j_idx[i_idx < j_idx]
    k_idx = k_idx[i_idx < j_idx]
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    if tau < 1e-6:
        soft_min = np.min(areas)
    else:
        scaled = -areas / tau
        max_scaled = np.max(scaled)
        soft_min = -tau * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    return -(soft_min / hull_area)

def _regular_polygon(n, radius=1.0, center=(0, 0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _optimize_from_start(start_pts, deadline, max_time_per_opt=10.0):
    """Optimize from a starting configuration using annealed soft-min."""
    best_score = _score(start_pts)
    best_pts = start_pts.copy()
    
    start_time = time.time()
    flat = start_pts.flatten().copy()
    
    # Anneal temperature from high to low
    taus = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for tau in taus:
        if time.time() > deadline - 5.0:
            break
        if time.time() - start_time > max_time_per_opt:
            break
        
        try:
            res = minimize(
                _soft_min_objective_fast,
                flat,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-8, 'gtol': 1e-6},
                timeout=min(10.0, max(1.0, deadline - time.time() - 5.0))
            )
            flat = res.x
            pts = flat.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    deadline = time.time() + 130.0  # Safety margin from 150s limit
    
    # Multiple diverse starting points
    starts = []
    
    # Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    
    # Regular 13-gon with slight perturbation
    rng = np.random.RandomState(42)
    for trial in range(3):
        base = _regular_polygon(N, radius=1.0)
        noise = rng.normal(0, 0.05, size=base.shape)
        starts.append(base + noise)
    
    # Points on ellipse
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    for rx, ry in [(1.5, 0.7), (0.7, 1.5), (1.2, 1.2)]:
        x = rx * np.cos(angles)
        y = ry * np.sin(angles)
        starts.append(np.column_stack([x, y]))
    
    # Random points in unit disk
    for trial in range(2):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Grid-like arrangement
    grid_starts = []
    for scale in [0.5, 0.8, 1.0]:
        # 4x4 grid minus one corner, plus adjustments
        xs = np.array([0, 1, 2, 3]) * scale
        ys = np.array([0, 1, 2, 3]) * scale
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])[:N]
        # Center it
        pts -= pts.mean(axis=0)
        grid_starts.append(pts)
    starts.extend(grid_starts)
    
    best_pts = starts[0]
    best_score = _score(best_pts)
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        try:
            opt_pts, opt_score = _optimize_from_start(start, deadline, max_time_per_opt=8.0)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass
    
    # Final local polish with very low tau on the best
    if time.time() < deadline - 5.0:
        try:
            flat = best_pts.flatten().copy()
            for tau in [0.0005, 0.0002, 0.0001]:
                if time.time() > deadline - 5.0:
                    break
                res = minimize(
                    _soft_min_objective_fast,
                    flat,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'maxfun': 50, 'ftol': 1e-10, 'gtol': 1e-8},
                    timeout=max(1.0, deadline - time.time() - 5.0)
                )
                flat = res.x
                pts = flat.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
        except Exception:
            pass
    
    return best_pts

N = 13
