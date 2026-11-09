import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _all_triangle_indices(n=13):
    """Precompute all C(n,3) triangle index triples."""
    return np.array(list(combinations(range(n), 3)), dtype=int)

def _triangle_areas(points, indices=None):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    if indices is None:
        indices = _all_triangle_indices(len(points))
    i, j, k = indices[:, 0], indices[:, 1], indices[:, 2]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _hull_area(points):
    """Return convex hull area, or 0 on failure."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        a_hull = _hull_area(points)
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = float(np.min(areas))
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _penalty_objective(x, penalty=10.0, smooth=1.0):
    """
    Objective for L-BFGS-B: maximize min triangle area / hull area.
    We minimize negative of that, with a penalty for degenerate configurations.
    
    Uses a smooth approximation: min(areas) ≈ -logsumexp(-areas/smooth)/n_triangles
    scaled appropriately, plus a large penalty if any area is near zero.
    """
    points = x.reshape(13, 2)
    if np.any(~np.isfinite(points)):
        return 1e6
    
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 1e6
    
    areas = _triangle_areas(points)
    a_min = float(np.min(areas))
    
    if a_min <= 1e-12:
        return 1e6
    
    # Smooth min approximation for gradient computation
    # -logsumexp(-areas/smooth) * smooth / n ≈ min(areas) when smooth is small
    n_tri = len(areas)
    logsumexp_val = np.log(np.sum(np.exp(-areas / smooth)))
    smooth_min = -smooth * logsumexp_val / n_tri * n_tri  # This gives approx -smooth*log(sum(exp(-a/s)))
    # Actually: -smooth * log(sum(exp(-a_i/smooth))) is a smooth upper bound on min(a_i)
    # Let me use a simpler approach: just use the actual min with a penalty
    
    score = a_min / a_hull
    
    # Add penalty for points that are too close together
    min_dist = np.min(areas)  # proxy for minimum distance
    penalty_term = penalty * np.exp(-min_dist / 0.01)
    
    return -score + penalty_term

def _penalty_objective_simple(x, smooth=0.01):
    """
    Simpler penalty objective: minimize -(min_area / hull_area) with smooth min.
    """
    points = x.reshape(13, 2)
    if np.any(~np.isfinite(points)):
        return 1e6
    
    a_hull = _hull_area(points)
    if a_hull <= 0:
        return 1e6
    
    areas = _triangle_areas(points)
    
    # Smooth minimum using log-sum-exp trick
    # min(a) ≈ -smooth * log(sum(exp(-a_i/smooth)))
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum_exp = max_a + smooth * np.log(np.sum(np.exp(-shifted / smooth)))
    smooth_min = -smooth * (np.log(np.sum(np.exp(-areas / smooth))))
    
    # Use the actual min for the objective value
    a_min = float(np.min(areas))
    if a_min <= 1e-12:
        return 1e6
    
    return -(a_min / a_hull)

def _optimize_lbfgsb(points, max_iter=500, timeout=120):
    """
    Optimize point positions using L-BFGS-B with a penalty-based objective.
    """
    x0 = points.flatten().astype(float)
    
    def obj(x):
        return _penalty_objective_simple(x, smooth=0.01)
    
    start_time = time.time()
    try:
        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 2000,
            }
        )
        elapsed = time.time() - start_time
        if result.x is not None:
            opt_points = result.x.reshape(13, 2)
            if np.all(np.isfinite(opt_points)):
                return opt_points
    except Exception:
        pass
    
    return points

def _local_refine(points, max_rounds=30, tol=1e-12):
    """
    Local refinement: for each point, try small perturbations in multiple directions.
    """
    pts = points.copy().astype(float)
    n = len(pts)
    best_score = _score(pts)
    indices = _all_triangle_indices(n)
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.7071, 0.7071], [-0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, -0.7071],
        [0.5, 0.866], [-0.5, 0.866], [0.866, 0.5], [0.866, -0.5],
        [-0.866, 0.5], [-0.866, -0.5], [0.5, -0.866], [-0.5, -0.866]
    ])
    
    for rnd in range(max_rounds):
        improved = False
        for i in range(n):
            for d in directions:
                for step in [0.02, 0.01, 0.005, 0.002]:
                    new_pts = pts.copy()
                    new_pts[i] += d * step
                    s = _score(new_pts)
                    if s > best_score + tol:
                        pts = new_pts
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return pts, best_score

def _regular_polygon(n=13):
    """Regular n-gon on unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _spiral_pattern(n=13, radius=1.0):
    """Archimedean spiral pattern."""
    t = np.linspace(0, 4 * np.pi, n)
    r = radius * t / (4 * np.pi)
    return np.column_stack((r * np.cos(t), r * np.sin(t)))

def _concentric_rings(n=13):
    """Points on concentric rings: 1 center + 6 inner + 6 outer."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Inner ring: 6 points
    for i in range(6):
        angle = i * 2 * np.pi / 6
        pts.append([0.4 * np.cos(angle), 0.4 * np.sin(angle)])
    # Outer ring: 6 points
    for i in range(6):
        angle = (i + 0.5) * 2 * np.pi / 6
        pts.append([0.9 * np.cos(angle), 0.9 * np.sin(angle)])
    return np.array(pts[:n])

def _halton_sequence(n=13, dim=2):
    """Halton sequence in 2D."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        pts.append([halton(i, 2), halton(i, 3)])
    # Center and scale
    pts = np.array(pts)
    pts = (pts - 0.5) * 2.0
    return pts

def _generate_initial_points(method='polygon', seed=42):
    """Generate diverse initial point sets."""
    n = 13
    rng = np.random.default_rng(seed)
    
    if method == 'polygon':
        return _regular_polygon(n)
    elif method == 'grid':
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        pts = np.array([[x, y] for x in xs for y in ys])
        pts = pts[:13]
        pts += rng.normal(0, 0.05, size=(13, 2))
        return pts
    elif method == 'random':
        pts = rng.uniform(-1, 1, size=(13, 2))
        return pts
    elif method == 'mixed':
        pts = _regular_polygon(n)
        pts += rng.normal(0, 0.1, size=(13, 2))
        return pts
    elif method == 'spiral':
        return _spiral_pattern(n)
    elif method == 'rings':
        return _concentric_rings(n)
    elif method == 'halton':
        return _halton_sequence(n)
    else:
        return _regular_polygon(n)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 570  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    methods = ['polygon', 'halton', 'rings', 'spiral', 'grid', 'mixed', 'random']
    seeds = [42, 123, 456]
    
    for method in methods:
        for seed in seeds:
            if time.time() > deadline - 90:
                break
            
            init_pts = _generate_initial_points(method, seed)
            init_score = _score(init_pts)
            
            if init_score <= 0:
                continue
            
            # Phase 1: L-BFGS-B optimization
            if time.time() < deadline - 60:
                lbfgs_pts = _optimize_lbfgsb(init_pts, max_iter=300, timeout=60)
                lbfgs_score = _score(lbfgs_pts)
                
                if lbfgs_score > best_score:
                    best_score = lbfgs_score
                    best_points = lbfgs_pts.copy()
            
            # Phase 2: Local refinement from best of (init, lbfgs)
            candidate = best_points if (best_points is not None and _score(best_points) >= init_score) else init_pts
            if time.time() < deadline - 30:
                refined_pts, refined_score = _local_refine(candidate, max_rounds=25)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_pts.copy()
        
        if time.time() > deadline - 90:
            break
    
    # Final refinement pass if time allows
    if best_points is not None and time.time() < deadline - 15:
        final_pts, final_score = _local_refine(best_points, max_rounds=40)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    # Fallback
    if best_points is None or _score(best_points) <= 0:
        best_points = _regular_polygon(n)
    
    return best_points
