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

DEADLINE = 135

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
    """Convert (b, c) barycentric coords to (x,y).
    barycentric: a=1-b-c, point = a*V0 + b*V1 + c*V2
    x = b*1.0 + c*0.5, y = c*S3H
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric_correct(pts):
    """Convert (x,y) to (b,c) barycentric coords where a=1-b-c.
    x = b + 0.5*c, y = c*S3H
    => c = y/S3H, b = x - 0.5*c
    """
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


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


def _min_triangle_area(pts):
    return np.min(_all_triangle_areas(pts))


def _soft_min_area(pts, beta):
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _soft_min_grad(pts, beta):
    """Gradient of soft_min_area with respect to points."""
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
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])

    np.add.at(grad, _T_I, grad_pi)
    np.add.at(grad, _T_J, grad_pj)
    np.add.at(grad, _T_K, grad_pk)

    return grad


def _barycentric_penalty(bc):
    """Penalty for barycentric coords (b,c) outside the triangle: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    pen = 0.0
    for i in range(N):
        if a[i] < 0:
            pen += a[i] ** 2 * 500.0
        if b[i] < 0:
            pen += b[i] ** 2 * 500.0
        if c[i] < 0:
            pen += c[i] ** 2 * 500.0
    return pen


def _soft_min_bary_grad(flat_bc, beta):
    """Objective and gradient in barycentric coords: -soft_min_area + penalty.
    Uses a projection approach: project barycentric coords to valid triangle.
    """
    bc = flat_bc.reshape(-1, 2)
    # Project to valid barycentric: b>=0, c>=0, b+c<=1
    b = np.maximum(bc[:, 0], 0.0)
    c = np.maximum(bc[:, 1], 0.0)
    s = b + c
    mask = s > 1.0
    b[mask] /= s[mask]
    c[mask] /= s[mask]
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    val = -_soft_min_area(pts, beta)
    grad_pts = _soft_min_grad(pts, beta)

    # Chain rule: x = b + 0.5*c, y = c*S3H
    # d(val)/db = -d(soft_min)/db = -grad_pts[:,0]
    # d(val)/dc = -d(soft_min)/dc = -(0.5*grad_pts[:,0] + S3H*grad_pts[:,1])
    grad_b = -grad_pts[:, 0]
    grad_c = -(0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1])
    grad_flat = np.column_stack([grad_b, grad_c]).flatten()
    return val, grad_flat


def _push_apart(pts, alpha=0.01):
    """Move points apart to increase minimum area (gradient-free)."""
    pts = pts.copy()
    for _ in range(5):
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


def _decoupled_local_refine(pts, beta, n_iter=20):
    """
    Structural change: decoupled local optimization.
    For each point, optimize its position while holding others fixed,
    maximizing the minimum area of triangles involving that point.
    This decouples the non-smooth min into per-point subproblems.
    """
    pts = pts.copy()
    for _ in range(n_iter):
        for i in range(N):
            # Find all triangles involving point i
            # Triangles where i is one of the three vertices
            mask = (_T_I == i) | (_T_J == i) | (_T_K == i)
            tri_areas = _all_triangle_areas(pts)
            relevant_areas = tri_areas[mask]

            if len(relevant_areas) == 0:
                continue

            # Current min area involving point i
            min_area_i = np.min(relevant_areas)

            # Optimize point i's position to maximize min(relevant_areas)
            # Use a small local search around current position
            best_pos = pts[i].copy()
            best_min = min_area_i

            # Try moving point i in small steps
            for trial in range(8):
                # Random perturbation
                scale = 0.02 * (1.0 - trial / 10.0)
                if trial == 0:
                    # First try the gradient direction
                    grad_pts = _soft_min_grad(pts, beta)
                    step = -grad_pts[i] * 0.01
                else:
                    step = np.random.default_rng(trial + int(time.time()*1000) % 10000).normal(0, scale, size=2)

                new_pos = pts[i] + step
                # Project into triangle
                y = np.clip(new_pos[1], 0.0, S3H)
                x_min = y / S3
                x_max = 1.0 - y / S3
                new_pos = np.array([np.clip(new_pos[0], x_min, x_max), y])

                pts[i] = new_pos
                new_min = np.min(_all_triangle_areas(pts)[mask])
                if new_min > best_min:
                    best_min = new_min
                    best_pos = new_pos.copy()
                else:
                    pts[i] = best_pos.copy()

            pts[i] = best_pos

    return _project_to_triangle(pts)


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
                de_obj, bounds, maxiter=50, popsize=20,
                seed=42, tol=1e-10, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Sobol-grid + local nudge strategy
    # Initialize on quasi-random low-discrepancy points for broad coverage
    from scipy.stats import qmc

    def _sobol_points_in_triangle(n):
        """Generate n quasi-random points in the equilateral triangle."""
        sampler = qmc.Sobol(d=2, scramble=True)
        raw = sampler.random(n)
        # Convert to barycentric: use (b, c) with b>=0, c>=0, b+c<=1
        # Use rejection-free mapping: u,v in [0,1], if u+v>1 reflect
        b = raw[:, 0].copy()
        c = raw[:, 1].copy()
        mask = b + c > 1.0
        b[mask] = 1.0 - b[mask]
        c[mask] = 1.0 - c[mask]
        # Map to cartesian
        x = b + 0.5 * c
        y = c * S3H
        return np.column_stack([x, y])

    # Generate multiple Sobol-based initializations
    n_starts = 30
    bases = [init.copy()]
    for k in range(20):
        sobol_pts = _sobol_points_in_triangle(N)
        # Add small jitter to break symmetry
        jitter = rng.normal(0, 0.03, size=(N, 2))
        bases.append(_project_to_triangle(sobol_pts + jitter))
    for _ in range(5):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())
        bases.append(_push_apart(best_pts, 0.01).copy())

    # Local nudge refinement: move points in smallest triangles
    def _local_nudge(pts, n_rounds=50, step_scale=0.02):
        """Move points involved in currently smallest triangles."""
        pts = pts.copy()
        for _ in range(n_rounds):
            if not _elapsed_ok(5):
                break
            areas = _all_triangle_areas(pts)
            # Find the 10 smallest triangles
            idx = np.argsort(areas)[:10]
            for tri_idx in idx:
                i, j, k = _TRIPLETS[tri_idx]
                # Try small moves for each of the 3 points
                for p_idx in [i, j, k]:
                    best_pos = pts[p_idx].copy()
                    best_area = areas[tri_idx]
                    for _trial in range(12):
                        # Random direction
                        angle = rng.uniform(0, 2 * np.pi)
                        step = step_scale * rng.uniform(0.3, 1.0)
                        dx = step * np.cos(angle)
                        dy = step * np.sin(angle)
                        new_pos = pts[p_idx] + np.array([dx, dy])
                        # Project
                        y_new = np.clip(new_pos[1], 0.0, S3H)
                        x_min = y_new / S3
                        x_max = 1.0 - y_new / S3
                        new_pos = np.array([np.clip(new_pos[0], x_min, x_max), y_new])
                        pts[p_idx] = new_pos
                        new_areas = _all_triangle_areas(pts)
                        new_min = np.min(new_areas)
                        if new_min > best_area:
                            best_area = new_min
                            best_pos = new_pos.copy()
                        else:
                            pts[p_idx] = best_pos.copy()
                    pts[p_idx] = best_pos
        return _project_to_triangle(pts)

    for s in range(n_starts):
        if not _elapsed_ok(25):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.03)
        _update_best(pts)

        # Apply local nudge refinement
        pts_nudged = _local_nudge(pts, n_rounds=30, step_scale=0.015)
        _update_best(pts_nudged)

        # Then do a few L-BFGS-B passes with moderate beta
        for beta in [100.0, 500.0, 2000.0]:
            if not _elapsed_ok(20):
                break
            try:
                bc = _cartesian_to_barycentric_correct(pts_nudged)
                flat_bc = bc.flatten()
                res = minimize(
                    lambda x, b=beta: _soft_min_bary_grad(x, b)[0],
                    flat_bc,
                    jac=lambda x, b=beta: _soft_min_bary_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
                )
                bc_opt = res.x.reshape(-1, 2)
                b = np.maximum(bc_opt[:, 0], 0.0)
                c = np.maximum(bc_opt[:, 1], 0.0)
                s_val = b + c
                mask = s_val > 1.0
                b[mask] /= s_val[mask]
                c[mask] /= s_val[mask]
                pts_opt = _barycentric_to_cartesian(np.column_stack([b, c]))
                _update_best(pts_opt)
                pts_nudged = pts_opt
            except Exception:
                pass

    # Phase C: Additional local nudge passes with smaller steps
    if best_pts is not None and _elapsed_ok(15):
        for step_scale in [0.01, 0.005, 0.002]:
            if not _elapsed_ok(10):
                break
            pts_nudged = _local_nudge(best_pts, n_rounds=40, step_scale=step_scale)
            _update_best(pts_nudged)
            best_pts = pts_nudged

    # Phase D: Nelder-Mead polish in barycentric coords (fixed projection)
    if best_pts is not None and _elapsed_ok(12):
        bc_best = _cartesian_to_barycentric_correct(best_pts)
        for scale in [0.03, 0.015, 0.008]:
            if not _elapsed_ok(8):
                break
            for _ in range(5):
                if not _elapsed_ok(6):
                    break
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                # Project to valid barycentric
                b = np.maximum(pert_bc[:, 0], 0.0)
                c = np.maximum(pert_bc[:, 1], 0.0)
                s_val = b + c
                mask = s_val > 1.0
                b[mask] /= s_val[mask]
                c[mask] /= s_val[mask]
                flat = np.column_stack([b, c]).flatten()

                def nm_obj(x):
                    bc_x = x.reshape(-1, 2)
                    bx = np.maximum(bc_x[:, 0], 0.0)
                    cx = np.maximum(bc_x[:, 1], 0.0)
                    s_val2 = bx + cx
                    mask2 = s_val2 > 1.0
                    bx[mask2] /= s_val2[mask2]
                    cx[mask2] /= s_val2[mask2]
                    px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                    return -_soft_min_area(px, 5000.0)

                try:
                    res = minimize(nm_obj, flat, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.maximum(bc_res[:, 0], 0.0)
                    c = np.maximum(bc_res[:, 1], 0.0)
                    s_val = b + c
                    mask = s_val > 1.0
                    b[mask] /= s_val[mask]
                    c[mask] /= s_val[mask]
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                except Exception:
                    pass

    # Phase E: Final fine polish with exact min (fixed projection)
    if best_pts is not None and _elapsed_ok(6):
        bc_best = _cartesian_to_barycentric_correct(best_pts)
        flat = bc_best.flatten()

        def final_nm(x):
            bc_x = x.reshape(-1, 2)
            bx = np.maximum(bc_x[:, 0], 0.0)
            cx = np.maximum(bc_x[:, 1], 0.0)
            s_val = bx + cx
            mask = s_val > 1.0
            bx[mask] /= s_val[mask]
            cx[mask] /= s_val[mask]
            px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
            return -_min_triangle_area(px)

        try:
            res = minimize(final_nm, flat, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            bc_res = res.x.reshape(-1, 2)
            b = np.maximum(bc_res[:, 0], 0.0)
            c = np.maximum(bc_res[:, 1], 0.0)
            s_val = b + c
            mask = s_val > 1.0
            b[mask] /= s_val[mask]
            c[mask] /= s_val[mask]
            p = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(p)
        except Exception:
            pass

    # Phase F: Final local nudge with very small steps
    if best_pts is not None and _elapsed_ok(3):
        pts_final = _local_nudge(best_pts, n_rounds=20, step_scale=0.001)
        _update_best(pts_final)

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return _project_to_triangle(pts)

    # Final safety: project and verify all points are inside
    result = _project_to_triangle(best_pts.copy())
    # Double-check: verify each point is inside the triangle
    for i in range(N):
        x, y = result[i]
        if y < -1e-10 or y > S3H + 1e-10:
            result[i] = [0.5, 0.0]
        x_min = y / S3
        x_max = 1.0 - y / S3
        if x < x_min - 1e-10 or x > x_max + 1e-10:
            result[i] = [(x_min + x_max) / 2, y]
    return result
# EVOLVE-BLOCK-END
