# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools


def _triangle_areas(pts):
    """Compute all triangle areas for 13 points (286 triangles)."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                areas[idx] = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                idx += 1
    return areas


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return np.inf


def _score(pts):
    """Compute A_min / A_hull."""
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    if a_min <= 0:
        return 0.0
    a_hull = _hull_area(pts)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 0.0
    return a_min / a_hull


def _soft_min_penalty(pts, temperature=0.05):
    """
    Smooth approximation of the minimum triangle area using soft-min.
    Returns -soft_min(areas) so we can minimize it.
    """
    areas = _triangle_areas(pts)
    min_val = np.min(areas)
    if min_val < 1e-10:
        return min_val
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_val - temperature * log_sum
    return -soft_min


def _objective(pts_flat, temperature=0.05):
    """Objective for L-BFGS-B: minimize -soft_min + penalty for hull area."""
    n = 13
    pts = pts_flat.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    penalty = _soft_min_penalty(pts, temperature)
    
    a_hull = _hull_area(pts)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 1e10
    
    alpha = 1.0
    return penalty + alpha * np.log(a_hull)


def _coordinate_descent(pts, max_iters=500, step=0.01, deadline=None):
    """
    Coordinate descent to refine points.
    For each point, try small perturbations in x and y.
    """
    n = 13
    pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                           (step/2, step/2), (-step/2, step/2),
                           (step/2, -step/2), (-step/2, -step/2)]:
                new_pts = pts.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                s = _score(new_pts)
                if s > best_score:
                    pts = new_pts
                    best_score = s
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return pts, best_score


def _hexagonal_seed():
    """Create a hexagonal lattice seed for 13 points."""
    pts = []
    pts.append([0.0, 0.0])
    r1 = 0.5
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 1.0
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)


def _ring_seed():
    """Create concentric ring seed."""
    pts = []
    pts.append([0.0, 0.0])
    r1 = 0.4
    for i in range(5):
        angle = i * 2 * np.pi / 5
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 0.9
    for i in range(7):
        angle = i * 2 * np.pi / 7
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)


def _spiral_seed():
    """Create a spiral seed."""
    pts = []
    n = 13
    for i in range(n):
        t = i / (n - 1)
        r = 0.1 + 0.9 * t
        angle = t * 4 * np.pi
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pts)


def _random_seed(rng):
    """Create a random seed."""
    return rng.random((13, 2))


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses multiple diverse seeds with L-BFGS-B optimization and coordinate descent refinement.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    seeds = [
        _hexagonal_seed(),
        _ring_seed(),
        _spiral_seed(),
    ]
    
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        seeds.append(_random_seed(rng))
    
    best_pts = None
    best_score = 0.0
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        seed = seed - np.mean(seed, axis=0)
        scale = np.max(np.abs(seed))
        if scale > 0:
            seed = seed / scale * 0.5
        
        try:
            res = minimize(
                _objective,
                seed.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < 1e8:
                pts = res.x.reshape(13, 2)
                if np.all(np.isfinite(pts)):
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
        except Exception:
            pass
        
        if best_pts is not None:
            refined_pts, refined_score = _coordinate_descent(
                best_pts, max_iters=200, step=0.05, deadline=deadline
            )
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    if best_pts is None:
        best_pts = _hexagonal_seed()
        best_pts = best_pts - np.mean(best_pts, axis=0)
        scale = np.max(np.abs(best_pts))
        if scale > 0:
            best_pts = best_pts / scale * 0.5
    
    if time.time() < deadline:
        best_pts, best_score = _coordinate_descent(
            best_pts, max_iters=100, step=0.02, deadline=deadline
        )
    
    return best_pts


# EVOLVE-BLOCK-END
