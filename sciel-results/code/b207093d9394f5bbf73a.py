"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 145  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _cartesian_to_barycentric(pts):
    """Convert (x,y) points to (a,b,c) barycentric coords with a+b+c=1."""
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc


def _barycentric_to_cartesian(abc):
    """Convert (a,b,c) barycentric coords to (x,y) cartesian points."""
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _objective_bary(x, mode='soft', beta=50.0):
    """
    Objective in barycentric coordinates.
    x: flattened barycentric coords (3*N)
    Returns negative of the metric (to be minimized).
    """
    abc = x.reshape(-1, 3)
    # Normalize barycentric coords
    abc = np.clip(abc, 0, 1)
    sums = abc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    abc = abc / sums
    pts = _barycentric_to_cartesian(abc)
    
    if mode == 'soft':
        val = -_soft_min_area(pts, beta)
    else:
        val = -_min_triangle_area(pts)
    return val


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Initial configuration: symmetric placement ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    # ---- Phase A: Differential Evolution in barycentric space ----
    if _elapsed_ok(40):
        try:
            def _de_obj(flat_bc):
                return _objective_bary(flat_bc, mode='hard')

            bounds_de = [(0, 1)] * (3 * N)
            de_result = differential_evolution(
                _de_obj,
                bounds_de,
                maxiter=30,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_result.fun < 0:
                abc = de_result.x.reshape(-1, 3)
                abc = np.clip(abc, 0, 1)
                sums = abc.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc = abc / sums
                de_pts = _barycentric_to_cartesian(abc)
                de_pts = _project_to_triangle(de_pts)
                _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Soft-min gradient optimization with annealing ----
    n_starts = 12
    bases_cart = [init.copy()]
    for _ in range(6):
        bases_cart.append(init + rng.normal(0, 0.2, size=(N, 2)))
    if best_pts is not None:
        bases_cart.append(best_pts.copy())

    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = bases_cart[s % len(bases_cart)]
        pts = base + rng.normal(0, 0.03, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)

        # Convert to barycentric for optimization
        abc = _cartesian_to_barycentric(pts)
        # Clip to valid barycentric
        for i in range(N):
            for j in range(3):
                abc[i, j] = max(0.0, min(1.0, abc[i, j]))
            s_val = abc[i].sum()
            if s_val > 0:
                abc[i] /= s_val

        # Annealing schedule for beta
        for beta in [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
            if not _elapsed_ok(20):
                break
            try:
                def _obj_soft(x, b=beta):
                    return _objective_bary(x, mode='soft', beta=b)

                res = minimize(
                    _obj_soft,
                    abc.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                abc_new = res.x.reshape(-1, 3)
                abc_new = np.clip(abc_new, 0, 1)
                sums = abc_new.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_new = abc_new / sums
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                _update_best(pts_new)
                abc = abc_new
            except Exception:
                pass

    # ---- Phase C: Nelder-Mead local refinement on hard objective ----
    if best_pts is not None and _elapsed_ok(30):
        abc_best = _cartesian_to_barycentric(best_pts)
        abc_best = np.clip(abc_best, 0, 1)
        sums = abc_best.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc_best = abc_best / sums

        def _obj_hard_nm(x):
            return _objective_bary(x, mode='hard')

        # Run multiple Nelder-Mead restarts with decreasing noise
        for noise_scale in [0.02, 0.01, 0.005, 0.002, 0.001]:
            nm_restarts = 20
            for i in range(nm_restarts):
                if not _elapsed_ok(5):
                    break
                
                # Perturb the best barycentric coords
                noise = rng.normal(0, noise_scale, size=(N, 3))
                x0 = abc_best.flatten() + noise.flatten()
                
                try:
                    res = minimize(
                        _obj_hard_nm,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    abc_res = res.x.reshape(-1, 3)
                    abc_res = np.clip(abc_res, 0, 1)
                    sums = abc_res.sum(axis=1, keepdims=True)
                    sums[sums == 0] = 1
                    abc_res = abc_res / sums
                    p = _barycentric_to_cartesian(abc_res)
                    p = _project_to_triangle(p)
                    _update_best(p)
                    
                    # Update abc_best if improved
                    if _min_triangle_area(p) > best_area - 1e-9:
                        abc_best = abc_res
                except Exception:
                    pass

    # ---- Phase D: Final fine-tuning with very small perturbations ----
    if best_pts is not None and _elapsed_ok(15):
        abc_best = _cartesian_to_barycentric(best_pts)
        abc_best = np.clip(abc_best, 0, 1)
        sums = abc_best.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc_best = abc_best / sums

        def _obj_hard_nm(x):
            return _objective_bary(x, mode='hard')

        for i in range(50):
            if not _elapsed_ok(3):
                break
            noise = rng.normal(0, 0.0005, size=(N, 3))
            x0 = abc_best.flatten() + noise.flatten()
            try:
                res = minimize(
                    _obj_hard_nm,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                abc_res = res.x.reshape(-1, 3)
                abc_res = np.clip(abc_res, 0, 1)
                sums = abc_res.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_res = abc_res / sums
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
                if _min_triangle_area(p) > best_area - 1e-9:
                    abc_best = abc_res
            except Exception:
                pass

    if best_pts is None:
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
            [1/3, S3H/3], [0.333, 0.0], [0.667, 0.0],
            [0.5, S3H/3], [0.2, 0.1]
        ])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
