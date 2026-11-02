import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all C(14,3)=364 triangles
_i, _j, _k = np.indices((N, N, N))
_mask = (_i < _j) & (_j < _k)
_IDX_I = _i[_mask]
_IDX_J = _j[_mask]
_IDX_K = _k[_mask]

def _compute_min_triangle_area(points):
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """Hard objective: maximize min triangle area after normalizing hull to unit area."""
    points = points_flat.reshape(N, 2)
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    scale = np.sqrt(1.0 / hull_area)
    points = points * scale
    min_area = _compute_min_triangle_area(points)
    return -min_area

def _soft_min_objective(points_flat, beta):
    """Smooth approximation of the hard objective using log-sum-exp soft-min."""
    points = points_flat.reshape(N, 2)
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    scale = np.sqrt(1.0 / hull_area)
    points = points * scale
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    soft_min = max_area + (1.0 / beta) * np.log(np.sum(exp_terms))
    return -soft_min

def _refine(x0, deadline):
    """Powell coarse search -> annealed soft-min L-BFGS-B -> Nelder-Mead true-objective polish."""
    x_current = x0
    # Coarse
    try:
        res = minimize(_objective, x_current, method='Powell',
                       options={'maxiter': 400, 'maxfev': 4000})
        if time.time() > deadline:
            return x_current
        x_current = res.x
    except Exception:
        pass
    # Annealed smooth refinement
    betas = [10, 50, 100, 300, 1000, 3000]
    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = minimize(lambda x, b=beta: _soft_min_objective(x, b), x_current,
                           method='L-BFGS-B',
                           options={'maxiter': 60, 'maxfev': 800, 'ftol': 1e-12, 'gtol': 1e-12})
            if time.time() > deadline:
                break
            x_current = res.x
        except Exception:
            pass
    # True-objective polish
    if time.time() < deadline - 3.0:
        try:
            res = minimize(_objective, x_current, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            x_current = res.x
        except Exception:
            pass
    return x_current

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s limit

    best_score = 0.0
    best_points = None

    rng = np.random.default_rng(42)

    inits = []
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # 2. Jittered regular 14-gon
    for jitter_std in [0.05, 0.1, 0.2]:
        jitter = rng.normal(0, jitter_std, (N, 2))
        inits.append(np.column_stack([np.cos(angles) + jitter[:, 0],
                                      np.sin(angles) + jitter[:, 1]]))
    # 3. Random points in circle (normalized to unit radius)
    for _ in range(12):
        pts = rng.random((N, 2)) * 2 - 1
        dists = np.linalg.norm(pts, axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts = pts / max_dist
        inits.append(pts)

    for init in inits:
        if time.time() > deadline - 5.0:
            break
        try:
            x_current = _refine(init.flatten(), deadline)
        except Exception:
            continue
        if time.time() > deadline:
            break
        pts = x_current.reshape(N, 2)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = np.array([[rng.random(), rng.random()] for _ in range(N)])

    return best_points
