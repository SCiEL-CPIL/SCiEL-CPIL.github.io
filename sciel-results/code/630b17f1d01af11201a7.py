import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _negative_score(x_flat):
    """Objective for optimization: -score."""
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _apply_affine(base_points, affine_params):
    """Apply affine transformation to base points.
    affine_params: [a, b, c, d] for matrix [[a, b], [c, d]]
    """
    a, b, c, d = affine_params
    M = np.array([[a, b], [c, d]])
    return base_points @ M.T


def _affine_score(affine_params, base_points):
    """Score after applying affine transform to base points."""
    points = _apply_affine(base_points, affine_params)
    if not np.all(np.isfinite(points)):
        return -1e10
    s = _score(points)
    return s if s > 0 else -1e10


def _negative_affine_score(x, base_points):
    """Negative score for affine optimization."""
    return -_affine_score(x, base_points)


def _grid_4x4_minus_corner():
    """4x4 grid minus top-right corner, centered."""
    pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            pts.append([i - 1.5, j - 1.5])
    return np.array(pts, dtype=float)


def _grid_4x4_minus_center():
    """4x4 grid minus center-ish point (remove one interior)."""
    pts = []
    for i in range(4):
        for j in range(4):
            # Remove point (1,2) to break symmetry slightly
            if i == 1 and j == 2:
                continue
            pts.append([i - 1.5, j - 1.5])
    return np.array(pts, dtype=float)


def _grid_3x3_plus_4():
    """3x3 grid (9 points) plus 4 extra points at midpoints of edges."""
    pts = []
    # 3x3 grid
    for i in range(3):
        for j in range(3):
            pts.append([i - 1.0, j - 1.0])
    # 4 edge midpoints of the 3x3 bounding box
    pts.append([0.0, 2.0])
    pts.append([0.0, -2.0])
    pts.append([2.0, 0.0])
    pts.append([-2.0, 0.0])
    return np.array(pts, dtype=float)


def _concentric_rings_template():
    """1 center + 5 inner ring + 7 outer ring."""
    center = [[0.0, 0.0]]
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_angles = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 7, endpoint=False)
    outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    return np.vstack([center, inner, outer])


def _hexagonal_lattice_template():
    """Hexagonal lattice: 1 center + 6 hex neighbors + 6 next ring."""
    pts = [[0.0, 0.0]]
    # First ring (hexagon)
    for k in range(6):
        angle = k * np.pi / 3
        pts.append([0.8 * np.cos(angle), 0.8 * np.sin(angle)])
    # Second ring (hexagon, offset)
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        pts.append([1.6 * np.cos(angle), 1.6 * np.sin(angle)])
    return np.array(pts, dtype=float)


def _optimize_affine_for_template(base_points, deadline):
    """Optimize affine transformation for a given template."""
    # Initial guess: identity
    x0 = np.array([1.0, 0.0, 0.0, 1.0])
    
    # Try multiple starting points
    starts = [
        np.array([1.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 0.3, 0.0, 1.0]),
        np.array([1.0, -0.3, 0.0, 1.0]),
        np.array([1.2, 0.0, 0.0, 0.8]),
        np.array([0.8, 0.0, 0.0, 1.2]),
        np.array([1.0, 0.5, 0.5, 1.0]),
        np.array([1.0, -0.5, -0.5, 1.0]),
    ]
    
    best_params = x0.copy()
    best_val = -1e10
    
    for start in starts:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(
                _negative_affine_score,
                start,
                args=(base_points,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            val = _affine_score(result.x, base_points)
            if val > best_val:
                best_val = val
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Also try DE in 4D for global search
    if time.time() < deadline - 10:
        try:
            bounds = [(-2.0, 2.0)] * 4
            de_result = differential_evolution(
                _negative_affine_score,
                bounds=bounds,
                args=(base_points,),
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-6,
                init='sobol'
            )
            val = _affine_score(de_result.x, base_points)
            if val > best_val:
                best_val = val
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    return best_params, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin against 120s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Phase 1: Initial Guesses
    _update_best(_regular_polygon(N, 1.0))
    
    center = np.array([[0.0, 0.0]])
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    _update_best(np.vstack([center, inner_pts, outer_pts]))
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.uniform(-1, 1, (N, 2))
        _update_best(pts)

    # Phase 2: Optimize affine transforms for grid templates
    templates = [
        _grid_4x4_minus_corner(),
        _grid_4x4_minus_center(),
        _grid_3x3_plus_4(),
        _concentric_rings_template(),
        _hexagonal_lattice_template(),
    ]
    
    for tmpl in templates:
        if time.time() > deadline - 10:
            break
        if len(tmpl) != N:
            continue
        try:
            params, val = _optimize_affine_for_template(tmpl, deadline)
            if val > 0:
                transformed = _apply_affine(tmpl, params)
                _update_best(transformed)
        except Exception:
            pass

    # Phase 3: Full 26D optimization from best point
    if best_points is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            _update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 4: DE in full 26D if time permits
    if time.time() < deadline - 20:
        try:
            bounds = [(-2.0, 2.0)] * (N * 2)
            de_result = differential_evolution(
                _negative_score,
                bounds=bounds,
                seed=42,
                maxiter=25,
                popsize=15,
                tol=1e-4,
                init='sobol'
            )
            _update_best(de_result.x.reshape(N, 2))
            
            # Polish
            if time.time() < deadline - 5:
                result2 = minimize(
                    _negative_score,
                    de_result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                _update_best(result2.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 5: More random restarts if time permits
    if time.time() < deadline - 15:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 5:
                break
            try:
                rng2 = np.random.default_rng(seed)
                pts = rng2.uniform(-1.5, 1.5, (N, 2))
                _update_best(pts)
                
                result = minimize(
                    _negative_score,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                _update_best(result.x.reshape(N, 2))
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
