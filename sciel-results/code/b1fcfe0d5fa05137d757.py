# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time


def _hex_vertices(cx, cy, d, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.array([cx + d * np.cos(angles), cy + d * np.sin(angles)]).T


def _lp_optimal_center(centers, outer_angle_deg):
    """
    Given inner centers (12,2) and outer hexagon angle, find the outer center
    that minimizes the required side length s.
    Returns (optimal_center, min_s).
    """
    a = np.deg2rad(outer_angle_deg)
    # Edge normals of outer hexagon
    normal_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.array([np.cos(normal_angles), np.sin(normal_angles)]).T  # (6,2)

    # All vertices of all inner hexagons (72, 2)
    all_verts = []
    for cx, cy in centers:
        all_verts.append(_hex_vertices(cx, cy, 1.0, 30.0))
    all_verts = np.vstack(all_verts)

    # LP: minimize t s.t. (v - c) . n_i <= t for all i, v
    # Variables: c_x, c_y, t  (3 vars)
    # Constraint: v.n - c.n <= t  =>  -c.n - t <= -v.n
    # c.n = c_x*n_x + c_y*n_y
    # So: -n_x*c_x - n_y*c_y - 1*t <= -v.n
    A_ub = []
    b_ub = []
    for v in all_verts:
        for n in normals:
            A_ub.append([-n[0], -n[1], -1.0])
            b_ub.append(-float(v @ n))
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    c0 = np.mean(centers, axis=0)
    res = linprog([0, 0, 1], A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (None, None)],
                  method='highs')
    if res.success:
        cx, cy, t = res.x
        center = np.array([cx, cy])
    else:
        center = c0.copy()
        # Fallback: compute t at centroid
        t = max(float((v - center) @ n) for v in all_verts for n in normals)

    s = 2.0 * t / np.sqrt(3.0)
    return center, s


def _separation_violation(centers, tol=1e-6):
    """
    Return total separation violation (0 if all pairs separated).
    Two hexagons are separated if for some edge normal of either,
    the projection intervals are more than tol apart (gap > 0 means separated).
    Violation = sum of max(0, -gap + tol) over all pairs/axes.
    """
    verts_list = [_hex_vertices(cx, cy, 1.0, 30.0) for cx, cy in centers]
    viol = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            v1 = verts_list[i]
            v2 = verts_list[j]
            for verts in (v1, v2):
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    norm = np.linalg.norm(edge)
                    if norm < 1e-12:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / norm
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    gap = max(np.min(proj1) - np.max(proj2),
                              np.min(proj2) - np.max(proj1))
                    if gap < tol:
                        viol += (tol - gap)
    return viol


def _objective(x, outer_angle_deg):
    """
    Objective for Nelder-Mead: x = flattened centers (24,)
    Returns s + penalty * separation_violation
    """
    centers = x.reshape(12, 2)
    center, s = _lp_optimal_center(centers, outer_angle_deg)
    # penalty for separation violations
    viol = _separation_violation(centers)
    return s + 10.0 * viol


def hexagon_packing_12():
    start_time = time.time()
    DEADLINE = 600.0

    # Initial 3x4 hexagonal grid of centers (flat-top, angle 30)
    scale = 1.001
    centers_init = []
    for r in range(3):
        for c in range(4):
            x = scale * (c * np.sqrt(3) + (r % 2) * (np.sqrt(3) / 2))
            y = scale * (r * 1.5)
            centers_init.append([x, y])
    centers_init = np.array(centers_init)
    centers_init -= np.mean(centers_init, axis=0)

    best_s = np.inf
    best_centers = centers_init.copy()
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0

    # Try a range of outer angles
    for outer_angle_deg in np.linspace(0, 30, 31):
        if time.time() - start_time > DEADLINE - 60:
            break

        # LP optimal center for initial configuration
        c0, s0 = _lp_optimal_center(centers_init, outer_angle_deg)
        x0 = centers_init.flatten()

        # Nelder-Mead refinement on inner positions
        def obj(x):
            return _objective(x, outer_angle_deg)

        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-7})
        except Exception:
            res = None

        if res is not None and res.x is not None:
            c_opt, s_opt = _lp_optimal_center(res.x.reshape(12, 2), outer_angle_deg)
            viol = _separation_violation(res.x.reshape(12, 2))
            if viol < 1e-4 and s_opt < best_s:
                best_s = s_opt
                best_centers = res.x.reshape(12, 2)
                best_center = c_opt
                best_angle = outer_angle_deg
        else:
            if s0 < best_s:
                best_s = s0
                best_centers = centers_init.copy()
                best_center = c0
                best_angle = outer_angle_deg

    # Final: recompute optimal center for best configuration
    best_center, best_s = _lp_optimal_center(best_centers, best_angle)

    # Build inner data: (12,3) rows (x, y, angle_deg)
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i] = [best_centers[i, 0], best_centers[i, 1], 30.0]

    # Outer data: (3,) (x, y, angle_deg), centered at best_center
    outer = np.array([best_center[0], best_center[1], best_angle])

    return inner, outer, best_s


# EVOLVE-BLOCK-END
