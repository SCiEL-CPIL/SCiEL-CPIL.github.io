import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 120.0
SAFETY_MARGIN = 8.0
START_TIME = time.time()

# Precompute all triangle index triples for vectorized min-area
_TRIPLE_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)], dtype=np.intp)

def _min_triangle_area(points):
    """Vectorized minimum triangle area over all C(N,3) triples."""
    p_i = points[_TRIPLE_IDX[:, 0]]
    p_j = points[_TRIPLE_IDX[:, 1]]
    p_k = points[_TRIPLE_IDX[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Objective to minimize: -min_triangle_area / hull_area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _concentric(ring_sizes, radii, phase=0.0):
    """Points on concentric circles with rotational symmetry."""
    pts = []
    for size, r in zip(ring_sizes, radii):
        if size == 0:
            continue
        ang = np.linspace(0, 2 * np.pi, size, endpoint=False) + phase
        pts.append(np.column_stack((r * np.cos(ang), r * np.sin(ang))))
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)

def _normalize(points):
    """Scale so convex hull area = 1 (scale invariance)."""
    a = _convex_hull_area(points)
    if a < 1e-12:
        return points
    return points * np.sqrt(1.0 / a)

def _local_polish(x0, start_time):
    """Nelder-Mead local refinement with time guard."""
    try:
        res = minimize(
            _score, x0, method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        if res.fun < _score(x0):
            return res.x
    except Exception:
        pass
    return x0

def construct():
    start_time = time.time()
    best_points = None
    best_score = -float('inf')

    def consider(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()

    # ---- Diverse initial configurations ----
    inits = []

    # Regular 13-gon
    inits.append(_regular_polygon(N))

    # Concentric ring arrangements (rotational symmetry breaks small-triangle issue)
    ring_configs = [
        ([1, 12], [0.0, 1.0]),
        ([5, 8], [0.0, 1.0]),
        ([4, 9], [0.0, 1.0]),
        ([7, 6], [0.0, 1.0]),
        ([6, 7], [0.0, 1.0]),
        ([3, 5, 5], [0.0, 0.55, 1.0]),
        ([2, 5, 6], [0.0, 0.5, 1.0]),
        ([1, 5, 7], [0.0, 0.5, 1.0]),
        ([4, 4, 5], [0.0, 0.55, 1.0]),
        ([1, 4, 8], [0.0, 0.5, 1.0]),
        ([2, 3, 8], [0.0, 0.45, 1.0]),
        ([1, 6, 6], [0.0, 0.55, 1.0]),
    ]
    for sizes, radii in ring_configs:
        for phase in (0.0, np.pi / 13.0):
            inits.append(_concentric(sizes, radii, phase=phase))

    # Jittered versions of the best ring configs to escape symmetries
    for sizes, radii in ring_configs[:6]:
        base = _concentric(sizes, radii)
        jittered = base + np.random.normal(0, 0.08, base.shape)
        inits.append(jittered)

    # ---- Phase 1: Global search via differential evolution (a few trials) ----
    bounds = [(-1.5, 1.5)] * (2 * N)
    for trial in range(3):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        try:
            de_res = differential_evolution(
                _score, bounds,
                seed=trial * 1000 + 7,
                maxiter=40,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='sobol'
            )
            x_de = de_res.x
            if _score(x_de) < best_score:
                best_score = _score(x_de)
                best_points = x_de.reshape(-1, 2).copy()
        except Exception:
            pass

    # ---- Phase 2: Local polish from each diverse start + DE result ----
    polish_starts = [init.flatten() for init in inits]
    if best_points is not None:
        polish_starts.append(best_points.flatten())

    for x0 in polish_starts:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        x_polished = _local_polish(x0, start_time)
        pts = x_polished.reshape(-1, 2)
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()

    # ---- Phase 3: Second round of local polish from the best, with tiny perturbations ----
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        for _ in range(8):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            pert = best_points + np.random.normal(0, 0.03, best_points.shape)
            x_polished = _local_polish(pert.flatten(), start_time)
            pts = x_polished.reshape(-1, 2)
            s = _score(pts)
            if s < best_score:
                best_score = s
                best_points = pts.copy()

    # ---- Final: normalize hull area to 1 (scale invariance, cleaner output) ----
    if best_points is None:
        best_points = _regular_polygon(N)
    else:
        best_points = _normalize(best_points)

    return best_points
