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

DEADLINE = 145

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _cartesian_to_barycentric(pts):
    """Convert (x,y) to (b, c) barycentric coords where a=1-b-c."""
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])


def _barycentric_to_cartesian(bc):
    """Convert (b, c) barycentric coords to (x,y)."""
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


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
    return max_a - log_sum / beta


def _soft_min_grad(pts, beta):
    """Compute gradient of soft_min_area with respect to points."""
    areas = _all_triangle_areas(pts)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2

    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp

    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0

    w = 0.5 * weights * sign_cross

    grad = np.zeros((N, 2))

    # da/dpi = 0.5 * sign(cross) * (-dy2, dx2)
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    # da/dpj = 0.5 * sign(cross) * (dy2, -dx2)
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    # da/dpk = 0.5 * sign(cross) * (-dy1, dx1)
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])

    np.add.at(grad, _T_I, grad_pi)
    np.add.at(grad, _T_J, grad_pj)
    np.add.at(grad, _T_K, grad_pk)

    return grad


def _barycentric_objective(flat_bc, beta):
    """Objective and gradient in barycentric coords: -soft_min_area."""
    bc = flat_bc.reshape(-1, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1 - b)
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    val = -_soft_min_area(pts, beta)

    # Compute gradient
    if beta > 0:
        grad_pts = _soft_min_grad(pts, beta)
        # Chain rule: barycentric to cartesian
        # x = b + 0.5*c, y = c*S3H
        # grad_b = grad_x, grad_c = 0.5*grad_x + S3H*grad_y
        grad_b = grad_pts[:, 0]
        grad_c = 0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1]
        grad_flat = np.column_stack([-grad_b, -grad_c]).flatten()
    else:
        grad_flat = np.zeros_like(flat_bc)

    return val, grad_flat


def _push_apart(pts, alpha=0.01, iterations=5):
    """Move points apart to increase minimum area (gradient-free)."""
    pts = pts.copy()
    for _ in range(iterations):
        disp = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist > 1e-8:
                    force = alpha / (dist * dist + 1e-6)
                    disp[i] += d / dist * force
                    disp[j] -= d / dist * force
        pts = _project_to_triangle(pts + disp)
    return pts


def _local_search_from(pts, max_iter=2000, step_scale=0.01):
    """Local search using coordinate perturbation and Nelder-Mead."""
    bc = _cartesian_to_barycentric(pts)
    flat = bc.flatten()

    def objective(x):
        bc_x = x.reshape(-1, 2)
        bx = np.clip(bc_x[:, 0], 0, 1)
        cx = np.clip(bc_x[:, 1], 0, 1 - bx)
        px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
        return -_min_triangle_area(px)

    try:
        res = minimize(objective, flat, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14})
        bc_res = res.x.reshape(-1, 2)
        b = np.clip(bc_res[:, 0], 0, 1)
        c = np.clip(bc_res[:, 1], 0, 1 - b)
        p = _barycentric_to_cartesian(np.column_stack([b, c]))
        return p
    except Exception:
        return pts


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

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = v * S3H
        return np.array([x, y])

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        return _project_to_triangle(pts)

    # Phase A: Differential Evolution (global)
    if _elapsed_ok(50):
        try:
            def de_obj(flat):
                pts = flat.reshape(-1, 2)
                return -_min_triangle_area(_project_to_triangle(pts))

            bounds = [(0, 1), (0, S3H)] * N
            de_res = differential_evolution(
                de_obj, bounds, maxiter=80, popsize=30,
                seed=42, tol=1e-10, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Multi-start barycentric L-BFGS-B with annealed beta
    n_starts = 60
    bases = [init.copy()]
    for _ in range(30):
        bases.append(_start_from_base(init, 0.25))
    for _ in range(20):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())
        bases.append(_push_apart(best_pts, 0.01).copy())

    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0, 1200.0, 3000.0, 8000.0]

    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.05)
        _update_best(pts)

        bc = _cartesian_to_barycentric(pts)
        flat_bc = bc.flatten()

        for beta in beta_schedule:
            if not _elapsed_ok(25):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _barycentric_objective(x, b)[0],
                    flat_bc,
                    jac=lambda x, b=beta: _barycentric_objective(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
                )
                bc_opt = res.x.reshape(-1, 2)
                b = np.clip(bc_opt[:, 0], 0, 1)
                c = np.clip(bc_opt[:, 1], 0, 1 - b)
                pts = _barycentric_to_cartesian(np.column_stack([b, c]))
                _update_best(pts)
                flat_bc = np.column_stack([b, c]).flatten()
            except Exception:
                pass

    # Phase C: Nelder-Mead polish in barycentric coords
    if best_pts is not None and _elapsed_ok(12):
        bc_best = _cartesian_to_barycentric(best_pts)
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(8):
                break
            for _ in range(10):
                if not _elapsed_ok(6):
                    break
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                b = np.clip(pert_bc[:, 0], 0, 1)
                c = np.clip(pert_bc[:, 1], 0, 1 - b)
                flat = np.column_stack([b, c]).flatten()

                def nm_obj(x):
                    bc_x = x.reshape(-1, 2)
                    bx = np.clip(bc_x[:, 0], 0, 1)
                    cx = np.clip(bc_x[:, 1], 0, 1 - bx)
                    px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                    return -_soft_min_area(px, 5000.0)

                try:
                    res = minimize(nm_obj, flat, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.clip(bc_res[:, 0], 0, 1)
                    c = np.clip(bc_res[:, 1], 0, 1 - b)
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                except Exception:
                    pass

    # Phase D: Final fine polish with exact min
    if best_pts is not None and _elapsed_ok(5):
        bc_best = _cartesian_to_barycentric(best_pts)
        flat = bc_best.flatten()

        def final_nm(x):
            bc_x = x.reshape(-1, 2)
            bx = np.clip(bc_x[:, 0], 0, 1)
            cx = np.clip(bc_x[:, 1], 0, 1 - bx)
            px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
            return -_min_triangle_area(px)

        try:
            res = minimize(final_nm, flat, method='Nelder-Mead',
                           options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16})
            bc_res = res.x.reshape(-1, 2)
            b = np.clip(bc_res[:, 0], 0, 1)
            c = np.clip(bc_res[:, 1], 0, 1 - b)
            p = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(p)
        except Exception:
            pass

    # Phase E: Additional local search
    if best_pts is not None and _elapsed_ok(3):
        bc_best = _cartesian_to_barycentric(best_pts)
        for scale in [0.002, 0.001, 0.0005]:
            if not _elapsed_ok(2):
                break
            for _ in range(8):
                if not _elapsed_ok(1):
                    break
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                b = np.clip(pert_bc[:, 0], 0, 1)
                c = np.clip(pert_bc[:, 1], 0, 1 - b)
                flat = np.column_stack([b, c]).flatten()

                def final2_nm(x):
                    bc_x = x.reshape(-1, 2)
                    bx = np.clip(bc_x[:, 0], 0, 1)
                    cx = np.clip(bc_x[:, 1], 0, 1 - bx)
                    px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                    return -_min_triangle_area(px)

                try:
                    res = minimize(final2_nm, flat, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.clip(bc_res[:, 0], 0, 1)
                    c = np.clip(bc_res[:, 1], 0, 1 - b)
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                except Exception:
                    pass

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
