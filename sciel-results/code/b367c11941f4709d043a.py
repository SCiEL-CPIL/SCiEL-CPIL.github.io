"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 148

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _cartesian_to_barycentric(pts):
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])


def _barycentric_to_cartesian(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])


def _project_to_triangle(pts):
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
    return np.min(np.abs(cross)) * 0.5


def _all_areas(pts):
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
    areas = _all_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    return max_a - np.log(np.sum(np.exp(shifted))) / beta


def _soft_min_grad(pts, beta):
    areas = _all_areas(pts)
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
    weights = exp_vals / np.sum(exp_vals)
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    w = 0.5 * weights * sign_cross
    grad = np.zeros((N, 2))
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])
    np.add.at(grad, _T_I, grad_pi)
    np.add.at(grad, _T_J, grad_pj)
    np.add.at(grad, _T_K, grad_pk)
    return grad


def _bary_obj(flat_bc, beta):
    bc = flat_bc.reshape(-1, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1 - b)
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    val = -_soft_min_area(pts, beta)
    if beta > 0:
        grad_pts = _soft_min_grad(pts, beta)
        grad_b = grad_pts[:, 0]
        grad_c = 0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1]
        grad_flat = np.column_stack([-grad_b, -grad_c]).flatten()
    else:
        grad_flat = np.zeros_like(flat_bc)
    return val, grad_flat


def _lem_heuristic(pts, max_iter=500, tol=1e-12):
    """Largest Empty Triangle heuristic: iteratively move the vertex of the
    minimum-area triangle toward the centroid of the convex hull of the
    remaining points to increase the minimum area."""
    pts = pts.copy()
    for it in range(max_iter):
        areas = _all_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        i, j, k = _TRIPLETS[min_idx]
        # Centroid of all points excluding i, j, k
        mask = np.ones(N, dtype=bool)
        mask[[i, j, k]] = False
        remaining = pts[mask]
        centroid = remaining.mean(axis=0)
        # For each of the three vertices, compute the new min area if we
        # move that vertex toward the centroid (parametric search)
        best_improve = 0.0
        best_move = None
        for v in [i, j, k]:
            # Try moving vertex v toward centroid
            # Parametrize: new_pos = pts[v] + t*(centroid - pts[v])
            # Find t in [0, 1] that maximizes the min area
            best_t = 0.0
            best_ma = min_area
            # Coarse search
            for t in np.linspace(0, 1, 21):
                trial = pts.copy()
                trial[v] = pts[v] + t * (centroid - pts[v])
                trial = _project_to_triangle(trial)
                ma = _min_triangle_area(trial)
                if ma > best_ma:
                    best_ma = ma
                    best_t = t
            # Fine search around best_t
            if best_t > 0:
                t_lo = max(0, best_t - 0.05)
                t_hi = min(1, best_t + 0.05)
                for t in np.linspace(t_lo, t_hi, 21):
                    trial = pts.copy()
                    trial[v] = pts[v] + t * (centroid - pts[v])
                    trial = _project_to_triangle(trial)
                    ma = _min_triangle_area(trial)
                    if ma > best_ma:
                        best_ma = ma
                        best_t = t
            improve = best_ma - min_area
            if improve > best_improve:
                best_improve = improve
                best_move = (v, best_t)
        if best_move is None or best_improve < tol:
            break
        v, t = best_move
        pts[v] = pts[v] + t * (centroid - pts[v])
        pts = _project_to_triangle(pts)
    return pts


def _gradient_step(pts, lr=0.005, max_iter=200):
    """Gradient ascent on soft-min area in barycentric coords."""
    bc = _cartesian_to_barycentric(pts)
    best_pts = pts.copy()
    best_area = _min_triangle_area(pts)
    for it in range(max_iter):
        areas = _all_areas(pts)
        beta = max(10.0, 50.0 / (best_area + 1e-8))
        grad_pts = _soft_min_grad(pts, beta)
        # Convert to barycentric gradient
        gb = grad_pts[:, 0]
        gc = 0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1]
        bc[:, 0] += lr * gb
        bc[:, 1] += lr * gc
        # Project to valid barycentric
        b = np.clip(bc[:, 0], 0, 1)
        c = np.clip(bc[:, 1], 0, 1 - b)
        bc = np.column_stack([b, c])
        pts = _barycentric_to_cartesian(bc)
        ma = _min_triangle_area(pts)
        if ma > best_area:
            best_area = ma
            best_pts = pts.copy()
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
    return best_pts


def _nm_polish(pts, max_iter=5000):
    bc = _cartesian_to_barycentric(pts)
    flat = bc.flatten()

    def obj(x):
        bc_x = x.reshape(-1, 2)
        bx = np.clip(bc_x[:, 0], 0, 1)
        cx = np.clip(bc_x[:, 1], 0, 1 - bx)
        px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
        return -_min_triangle_area(px)

    try:
        res = minimize(obj, flat, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-13, 'fatol': 1e-15})
        bc_res = res.x.reshape(-1, 2)
        b = np.clip(bc_res[:, 0], 0, 1)
        c = np.clip(bc_res[:, 1], 0, 1 - b)
        return _barycentric_to_cartesian(np.column_stack([b, c]))
    except Exception:
        return pts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        return np.array([u + 0.5 * v, v * S3H])

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1.0/3, S3H/3],
        [1.0/3, 0.0],
        [2.0/3, 0.0],
        [0.5, S3H/3],
        [0.5, 2*S3H/3],
    ], dtype=float)
    _update_best(init)

    # Apply LEM heuristic to initial
    lem_init = _lem_heuristic(init, max_iter=300)
    _update_best(lem_init)

    # Phase A: Differential Evolution (global)
    if _elapsed_ok(45):
        try:
            def de_obj(flat):
                pts = flat.reshape(-1, 2)
                return -_min_triangle_area(_project_to_triangle(pts))

            bounds = [(0, 1), (0, S3H)] * N
            de_res = differential_evolution(
                de_obj, bounds, maxiter=60, popsize=25,
                seed=42, tol=1e-10, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
            # Apply LEM to DE result
            de_lem = _lem_heuristic(de_pts, max_iter=200)
            _update_best(de_lem)
        except Exception:
            pass

    # Phase B: Multi-start with LEM + gradient + L-BFGS-B
    n_starts = 40
    bases = [init.copy(), lem_init.copy()]
    for _ in range(15):
        bases.append(init + rng.normal(0, 0.2, size=(N, 2)))
    for _ in range(10):
        bases.append(np.array([_random_point_in_triangle() for _ in range(N)]))
    if best_pts is not None:
        bases.append(best_pts.copy())
        bases.append(_lem_heuristic(best_pts, max_iter=100).copy())

    beta_schedule = [50.0, 200.0, 800.0, 3000.0]

    for s in range(n_starts):
        if not _elapsed_ok(20):
            break
        base = bases[s % len(bases)]
        pts = _project_to_triangle(base + rng.normal(0, 0.03, size=(N, 2)))

        # Apply LEM heuristic
        pts = _lem_heuristic(pts, max_iter=200)
        _update_best(pts)

        # Gradient step
        pts = _gradient_step(pts, lr=0.003, max_iter=100)
        _update_best(pts)

        # L-BFGS-B with soft-min
        bc = _cartesian_to_barycentric(pts)
        flat_bc = bc.flatten()
        for beta in beta_schedule:
            if not _elapsed_ok(15):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _bary_obj(x, b)[0],
                    flat_bc,
                    jac=lambda x, b=beta: _bary_obj(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                bc_opt = res.x.reshape(-1, 2)
                b = np.clip(bc_opt[:, 0], 0, 1)
                c = np.clip(bc_opt[:, 1], 0, 1 - b)
                pts = _barycentric_to_cartesian(np.column_stack([b, c]))
                _update_best(pts)
                flat_bc = np.column_stack([b, c]).flatten()
            except Exception:
                pass

        # Final LEM on the result
        pts = _lem_heuristic(pts, max_iter=150)
        _update_best(pts)

    # Phase C: Nelder-Mead polish
    if best_pts is not None and _elapsed_ok(10):
        for scale in [0.02, 0.01, 0.005, 0.002]:
            if not _elapsed_ok(6):
                break
            for _ in range(8):
                if not _elapsed_ok(4):
                    break
                bc_best = _cartesian_to_barycentric(best_pts)
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                b = np.clip(pert_bc[:, 0], 0, 1)
                c = np.clip(pert_bc[:, 1], 0, 1 - b)
                p = _barycentric_to_cartesian(np.column_stack([b, c]))
                p = _nm_polish(p, max_iter=3000)
                _update_best(p)

    # Phase D: Final fine polish
    if best_pts is not None and _elapsed_ok(5):
        p = _nm_polish(best_pts, max_iter=10000)
        _update_best(p)
        # One more LEM pass
        p = _lem_heuristic(p, max_iter=200)
        _update_best(p)
        p = _nm_polish(p, max_iter=5000)
        _update_best(p)

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    return _project_to_triangle(best_pts)
