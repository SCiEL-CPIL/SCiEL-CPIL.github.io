"""Heilbronn-11: maximize the smallest triangle area among 11 points in a unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _barycentric_to_cartesian(bary):
    """Convert barycentric (u, v) with w=1-u-v to Cartesian in the unit equilateral triangle."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v >= 0."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])


def _project_to_triangle(pts):
    """Project points into the triangle (Cartesian)."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
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
    """Compute all triangle areas."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    """Differentiable soft minimum of triangle areas."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _objective_bary(x, beta):
    """Objective in barycentric coords: negative soft-min area.
    x has shape (2*N,): first N are u, next N are v."""
    u = x[0:N]
    v = x[N:2*N]
    # Enforce barycentric constraints: u>=0, v>=0, u+v<=1
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    pts = _barycentric_to_cartesian(np.column_stack([u, v]))
    return -_soft_min_area(pts, beta)


def _push_apart(pts):
    """Gradient-like push: for the minimum triangle, push its vertices apart."""
    areas = _all_triangle_areas(pts)
    min_idx = np.argmin(areas)
    i, j, k = _TRIPLETS[min_idx]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    
    # Compute outward normal directions from the centroid of the min triangle
    centroid = (pi + pj + pk) / 3.0
    
    # Direction from centroid to each vertex, normalized
    dirs = np.zeros((N, 2))
    for idx, p in [(i, pi), (j, pj), (k, pk)]:
        d = p - centroid
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            dirs[idx] = d / norm
    
    return dirs


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
        return pts

    # ---- Initial configurations ----
    # Symmetric starting point: 3 corners, 3 edge midpoints, 1 centroid, 2 more
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/6, S3H/6],
        [5/6, S3H/6],
        [0.5, S3H/3],
        [0.5, 2*S3H/3],
    ], dtype=float)
    _update_best(init)

    # Convert to barycentric for optimization
    bary_init = _cartesian_to_barycentric(init)

    # ---- Phase A: Differential Evolution for global exploration ----
    if _elapsed_ok(60):
        # Bounds in barycentric space
        bounds = [(0, 1)] * N + [(0, 1)] * N

        def de_objective(x):
            return _objective_bary(x, 100.0)

        try:
            de_res = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=50, popsize=20,
                tol=1e-10, mutation=(0.5, 1.5), recombination=0.9,
                polish=True, updating='immediate'
            )
            if de_res.success or _elapsed_ok(30):
                u = np.clip(de_res.x[0:N], 0, 1)
                v = np.clip(de_res.x[N:2*N], 0, 1 - u)
                pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                _update_best(pts)
        except Exception:
            pass

    # ---- Phase B: Multi-start L-BFGS-B with annealed beta ----
    n_starts = 30
    bases = [init]
    # Add perturbed versions of the best found so far
    if best_pts is not None:
        for _ in range(5):
            bases.append(_project_to_triangle(best_pts + rng.normal(0, 0.15, size=(N, 2))))
    # Random starts
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bases.append(_barycentric_to_cartesian(np.column_stack([u, v])))

    for s in range(n_starts):
        if not _elapsed_ok(20):
            break
        base = bases[s % len(bases)]
        bary = _cartesian_to_barycentric(base)
        x0 = np.concatenate([bary[:, 0], bary[:, 1]])

        for beta in [20.0, 50.0, 150.0, 400.0, 1000.0, 3000.0]:
            if not _elapsed_ok(15):
                break
            try:
                res = minimize(_objective_bary, x0, method='L-BFGS-B',
                               args=(beta,),
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11})
                u = np.clip(res.x[0:N], 0, 1)
                v = np.clip(res.x[N:2*N], 0, 1 - u)
                pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                _update_best(pts)
                # Update x0 for next beta
                x0 = np.concatenate([u, v])
            except Exception:
                pass

    # ---- Phase C: Local Nelder-Mead polish on best configuration ----
    if best_pts is not None and _elapsed_ok(10):
        for scale in [0.03, 0.015, 0.007, 0.003]:
            if not _elapsed_ok(6):
                break
            for _ in range(6):
                if not _elapsed_ok(5):
                    break
                cand = _project_to_triangle(best_pts + rng.normal(0, scale, size=(N, 2)))
                bary_cand = _cartesian_to_barycentric(cand)
                x0 = np.concatenate([bary_cand[:, 0], bary_cand[:, 1]])

                def nm_obj(x):
                    u = x[0:N]
                    v = x[N:2*N]
                    u = np.clip(u, 0, 1)
                    v = np.clip(v, 0, 1 - u)
                    pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                    return -_soft_min_area(pts, 1500.0)

                try:
                    res = minimize(nm_obj, x0, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                    u = np.clip(res.x[0:N], 0, 1)
                    v = np.clip(res.x[N:2*N], 0, 1 - u)
                    pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                    _update_best(pts)
                except Exception:
                    pass

    # ---- Phase D: Iterative push-apart refinement ----
    if best_pts is not None and _elapsed_ok(5):
        pts = best_pts.copy()
        for _ in range(200):
            if not _elapsed_ok(3):
                break
            dirs = _push_apart(pts)
            step = 0.005
            new_pts = pts + step * dirs
            new_pts = _project_to_triangle(new_pts)
            if _min_triangle_area(new_pts) > _min_triangle_area(pts):
                pts = new_pts
                _update_best(pts)
            else:
                step *= 0.5
                if step < 1e-6:
                    break

    # ---- Final fine polish ----
    if best_pts is not None and _elapsed_ok(4):
        bary_best = _cartesian_to_barycentric(best_pts)
        x0 = np.concatenate([bary_best[:, 0], bary_best[:, 1]])

        def final_obj(x):
            u = x[0:N]
            v = x[N:2*N]
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1 - u)
            pts = _barycentric_to_cartesian(np.column_stack([u, v]))
            return -_soft_min_area(pts, 5000.0)

        try:
            res = minimize(final_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            u = np.clip(res.x[0:N], 0, 1)
            v = np.clip(res.x[N:2*N], 0, 1 - u)
            pts = _barycentric_to_cartesian(np.column_stack([u, v]))
            _update_best(pts)
        except Exception:
            pass

    if best_pts is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
