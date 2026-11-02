"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

def _cartesian_to_barycentric(pts):
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
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _min_triangle_area(pts):
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
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _barycentric_penalty(abc):
    neg = np.minimum(abc, 0.0)
    pos = np.maximum(abc - 1.0, 0.0)
    penalty = 500.0 * (np.sum(neg ** 2) + np.sum(pos ** 2))
    sums = abc.sum(axis=1)
    penalty += 500.0 * np.sum((sums - 1.0) ** 2)
    return penalty

def _objective_bary(flat, beta):
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _soft_min_area(pts, beta)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _objective_bary_hard(flat):
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _min_triangle_area(pts)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _epigraph_objective(flat, t_init=0.0):
    """Epigraph formulation: minimize -t subject to all areas >= t.
    We use a smooth approximation of the constraint via soft-min.
    Actually, let's use a penalty approach with the min area directly."""
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _min_triangle_area(pts)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _push_apart(pts, alpha=1.0):
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = dist > 1e-10
    dist_safe = np.where(mask, dist, 1.0)
    force_mag = np.where(mask, alpha / (dist_safe ** 2), 0.0)
    disp = np.sum(force_mag[:, :, np.newaxis] * diff / dist_safe[:, :, np.newaxis], axis=1)
    return pts + disp

def _project_to_triangle(pts):
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _normalize_barycentric(abc):
    abc = np.clip(abc, 0.0, 1.0)
    sums = abc.sum(axis=1, keepdims=True)
    sums = np.where(sums > 1e-12, sums, 1.0)
    abc = abc / sums
    return abc

def construct():
    start_time = time.time()
    rng = np.random.default_rng(777)

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

    # Initial configurations
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.2, S3H/6],
        [0.8, S3H/6],
        [0.5, 2*S3H/3],
        [0.35, S3H/2.5],
    ], dtype=float)
    _update_best(init)

    def _make_ring_config(scale=0.5):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [0.5, S3H]
        pts[3] = [0.5, 0.0]
        pts[4] = [0.75, S3H / 2]
        pts[5] = [0.25, S3H / 2]
        cx, cy = 1/3, S3H/3
        for i in range(3, 8):
            angle = 2 * np.pi * (i - 3) / 5
            r = scale * 0.3 if i < 6 else scale * 0.5
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle) * 0.8
            pts[i] = [x, y]
        return pts

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        a = 1 - u - v
        x = a * 0 + u * 1 + v * 0.5
        y = a * 0 + u * 0 + v * S3H
        return np.array([x, y])

    bases_cart = [init.copy()]
    bases_cart.append(_make_ring_config(0.6))
    bases_cart.append(_make_ring_config(0.4))
    for _ in range(10):
        bases_cart.append(init + rng.normal(0, 0.15, size=(N, 2)))
    for _ in range(6):
        bases_cart.append(np.array([_random_point_in_triangle() for _ in range(N)]))
    if best_pts is not None:
        bases_cart.append(best_pts.copy())

    # Phase A: Differential Evolution in barycentric (b,c) space
    if _elapsed_ok(50):
        try:
            def _de_obj(flat_bc):
                bc = flat_bc.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                neg = np.minimum(abc, 0.0)
                pos = np.maximum(abc - 1.0, 0.0)
                pen = 500.0 * (np.sum(neg ** 2) + np.sum(pos ** 2))
                pts = _barycentric_to_cartesian(abc)
                ma = _min_triangle_area(pts)
                return -ma + pen

            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj,
                bounds_de,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.4, 1.2),
                recombination=0.8
            )
            bc = de_result.x.reshape(-1, 2)
            b = bc[:, 0]
            c = bc[:, 1]
            a = 1.0 - b - c
            abc = np.column_stack([a, b, c])
            abc = _normalize_barycentric(abc)
            de_pts = _barycentric_to_cartesian(abc)
            de_pts = _project_to_triangle(de_pts)
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Soft-min gradient optimization with adaptive beta
    n_starts = min(12, len(bases_cart))
    for s in range(n_starts):
        if not _elapsed_ok(45):
            break
        base = bases_cart[s].copy()
        pts = base + rng.normal(0, 0.02, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)

        abc = _cartesian_to_barycentric(pts)
        abc = _normalize_barycentric(abc)

        betas = [30.0, 100.0, 300.0, 1000.0]
        for beta in betas:
            if not _elapsed_ok(40):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_bary(x, b),
                    abc.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                abc_new = _normalize_barycentric(res.x.reshape(-1, 3))
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                _update_best(pts_new)
                abc = abc_new
            except Exception:
                pass

    # Phase C: Push-apart + Nelder-Mead
    if best_pts is not None and _elapsed_ok(30):
        for iteration in range(25):
            if not _elapsed_ok(15):
                break

            cand = best_pts.copy()
            for alpha in [0.008, 0.004, 0.002, 0.001]:
                cand = _push_apart(cand, alpha=alpha)
                cand = _project_to_triangle(cand)
            _update_best(cand)

            try:
                abc0 = _cartesian_to_barycentric(best_pts)
                abc0 = _normalize_barycentric(abc0)
                res = minimize(
                    _objective_bary_hard,
                    abc0.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                abc_res = _normalize_barycentric(res.x.reshape(-1, 3))
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    # Phase D: High-precision polish
    if best_pts is not None and _elapsed_ok(12):
        for _ in range(5):
            if not _elapsed_ok(6):
                break
            try:
                abc = _cartesian_to_barycentric(best_pts)
                abc = _normalize_barycentric(abc)
                res = minimize(
                    _objective_bary_hard,
                    abc.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                abc_res = _normalize_barycentric(res.x.reshape(-1, 3))
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    # Phase E: Final local search with multiple perturbations
    if best_pts is not None and _elapsed_ok(8):
        for _ in range(8):
            if not _elapsed_ok(4):
                break
            perturbed = best_pts + rng.normal(0, 0.005, size=(N, 2))
            perturbed = _project_to_triangle(perturbed)
            _update_best(perturbed)
            try:
                abc = _cartesian_to_barycentric(best_pts)
                abc = _normalize_barycentric(abc)
                res = minimize(
                    _objective_bary_hard,
                    abc.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                abc_res = _normalize_barycentric(res.x.reshape(-1, 3))
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
