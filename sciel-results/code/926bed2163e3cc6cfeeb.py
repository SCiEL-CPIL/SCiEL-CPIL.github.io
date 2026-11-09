import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

# Precompute triple indices for vectorized area calculation
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

def _compute_min_area(points):
    """Vectorized minimum triangle area for all triples."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.min(0.5 * np.abs(cross))

def _hull_area_fast(points):
    """Compute convex hull area using Andrew's monotone chain (fast, no SciPy overhead)."""
    pts = np.array(points, dtype=np.float64)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    n = len(pts)
    if n < 3:
        return 0.0
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for i in range(n):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pts[i]) <= 1e-12:
            lower.pop()
        lower.append(pts[i])
    
    upper = []
    for i in range(n - 1, -1, -1):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pts[i]) <= 1e-12:
            upper.pop()
        upper.append(pts[i])
    
    hull = np.vstack([lower[:-1], upper[:-1]])
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        return _hull_area_fast(points)
    except Exception:
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

def get_score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    ma = _compute_min_area(points)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.column_stack((x, y))

def _affine_transform(points, M, t):
    """Apply affine transformation: points @ M.T + t"""
    return points @ M.T + t

def _affine_score(params):
    """
    Score for affine-transformed regular 14-gon.
    params: [m00, m01, m10, m11, tx, ty]
    M = [[m00, m01], [m10, m11]], t = [tx, ty]
    """
    M = np.array([[params[0], params[1]],
                  [params[2], params[3]]])
    t = np.array([params[4], params[5]])
    base = _regular_polygon(N)
    pts = _affine_transform(base, M, t)
    return get_score(pts)

def _optimize_affine(deadline):
    """Optimize affine transformation of regular 14-gon."""
    # Initial: identity transform
    init = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    best_params = init.copy()
    best_score = _affine_score(init)
    
    # Global search with differential evolution (tighter bounds, fewer iterations)
    bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0), (-1.0, 1.0)]
    
    try:
        if time.time() < deadline - 20.0:
            de_result = differential_evolution(
                lambda p: -_affine_score(p),
                bounds,
                seed=42,
                maxiter=60,
                tol=1e-6,
                popsize=12,
                workers=-1
            )
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_params = de_result.x
    except Exception:
        pass
    
    # Local refinement with Powell
    try:
        if time.time() < deadline - 8.0:
            result = minimize(
                lambda p: -_affine_score(p),
                best_params,
                method='Powell',
                options={'maxiter': 150, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            powell_score = -result.fun
            if powell_score > best_score:
                best_score = powell_score
                best_params = result.x
    except Exception:
        pass
    
    M = np.array([[best_params[0], best_params[1]],
                  [best_params[2], best_params[3]]])
    t = np.array([best_params[4], best_params[5]])
    base = _regular_polygon(N)
    pts = _affine_transform(base, M, t)
    
    return pts, best_score

def _make_concentric_rings(n_rings, ring_sizes, radii, angle_offsets):
    """
    Create points in concentric rings.
    n_rings: number of rings
    ring_sizes: list of ints, number of points in each ring (must sum to N)
    radii: list of floats, radius of each ring
    angle_offsets: list of floats, angular offset for each ring
    """
    points = []
    for i in range(n_rings):
        n_pts = ring_sizes[i]
        r = radii[i]
        offset = angle_offsets[i]
        if n_pts == 1:
            angles = np.array([0.0])
        else:
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points.append(np.column_stack((x, y)))
    return np.vstack(points)

def _optimize_rings(ring_sizes, initial_radii, initial_offsets, deadline):
    """
    Optimize radii and angle offsets for a given ring structure.
    Returns (best_points, best_score).
    """
    n_rings = len(ring_sizes)
    
    def _eval_ring_params(params):
        radii = params[:n_rings]
        offsets = params[n_rings:]
        # Ensure radii are positive and sorted (inner to outer)
        radii = np.maximum(radii, 0.01)
        radii = np.sort(radii)
        # Normalize so outermost ring is at radius 1
        max_r = radii[-1]
        if max_r < 1e-10:
            return 0.0
        radii = radii / max_r
        points = _make_concentric_rings(n_rings, ring_sizes, radii, offsets)
        return get_score(points)
    
    # Initial point
    init_params = np.concatenate([initial_radii, initial_offsets])
    best_params = init_params.copy()
    best_score = _eval_ring_params(init_params)
    
    # Use differential evolution for global search (reduced iterations)
    bounds = [(0.01, 1.0)] * n_rings + [(0, 2*np.pi)] * n_rings
    
    try:
        if time.time() < deadline - 15.0:
            de_result = differential_evolution(
                lambda p: -_eval_ring_params(p),
                bounds,
                seed=42,
                maxiter=40,
                tol=1e-5,
                popsize=10,
                workers=-1
            )
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_params = de_result.x
    except Exception:
        pass
    
    # Local refinement with Powell
    try:
        if time.time() < deadline - 8.0:
            result = minimize(
                lambda p: -_eval_ring_params(p),
                best_params,
                method='Powell',
                options={'maxiter': 80, 'xtol': 1e-7, 'ftol': 1e-7}
            )
            powell_score = -result.fun
            if powell_score > best_score:
                best_score = powell_score
                best_params = result.x
    except Exception:
        pass
    
    # Get final points
    radii = np.maximum(best_params[:n_rings], 0.01)
    radii = np.sort(radii)
    max_r = radii[-1]
    if max_r < 1e-10:
        max_r = 1.0
    radii = radii / max_r
    offsets = best_params[n_rings:]
    points = _make_concentric_rings(n_rings, ring_sizes, radii, offsets)
    
    return points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_points = None
    best_score = 0.0
    
    # 1. Initialize with regular polygon
    regular_points = _regular_polygon(N)
    regular_score = get_score(regular_points)
    best_score = regular_score
    best_points = regular_points.copy()
    
    # 2. Try affine-transformed regular polygon (6 parameters - fast)
    try:
        aff_pts, aff_score = _optimize_affine(deadline)
        if aff_score > best_score:
            best_score = aff_score
            best_points = aff_pts.copy()
    except Exception:
        pass
    
    # 3. Try selected concentric ring structures (most promising ones)
    # Focus on structures likely to beat regular polygon
    ring_structures = [
        [7, 7],
        [8, 6],
        [6, 5, 3],
        [7, 4, 3],
        [5, 5, 4],
        [4, 4, 3, 3],
        [6, 4, 4],
        [5, 4, 5],
    ]
    
    for ring_sizes in ring_structures:
        if time.time() > deadline - 25.0:
            break
        
        n_rings = len(ring_sizes)
        
        # Initial radii: evenly spaced from 0.3 to 1.0
        if n_rings == 1:
            initial_radii = [1.0]
        else:
            initial_radii = np.linspace(0.3, 1.0, n_rings).tolist()
        
        # Initial offsets: staggered for better coverage
        initial_offsets = [i * 2*np.pi / max(sum(ring_sizes), 1) * ring_sizes[0] / 2.0 for i in range(n_rings)]
        
        try:
            points, score = _optimize_rings(ring_sizes, initial_radii, initial_offsets, deadline)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    # 4. 2D local refinement from best result
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(
                lambda x: -get_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 150, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # 5. Final check and return
    if best_points is None:
        best_points = regular_points

    return best_points
