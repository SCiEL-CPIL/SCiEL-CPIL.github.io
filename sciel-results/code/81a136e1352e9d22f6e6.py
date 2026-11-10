# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 600.0
_EPS = 1e-6
_TOL = 1e-9


def _hex_verts(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(ang), cy + s * np.sin(ang)])


def _normals_of_hex(verts):
    n = np.empty((6, 2))
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        d = p2 - p1
        n[i] = np.array([d[1], -d[0]])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-300)
    return n


def _pairwise_gap(v1, v2):
    """Max over 12 edge normals of the minimum positive separation.
    Positive => separated (no overlap)."""
    n1 = _normals_of_hex(v1)
    n2 = _normals_of_hex(v2)
    p1 = v1 @ n1.T   # (6,6)
    p2 = v2 @ n1.T
    g = np.minimum(
        np.max(p2, axis=0) - np.min(p1, axis=0),
        np.min(p2, axis=0) - np.max(p1, axis=0),
    )
    p1b = v1 @ n2.T
    p2b = v2 @ n2.T
    g2 = np.minimum(
        np.max(p2b, axis=0) - np.min(p1b, axis=0),
        np.min(p2b, axis=0) - np.max(p1b, axis=0),
    )
    return float(max(g.max(), g2.max()))


def _min_gap(verts_list):
    m = 0.0
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            m = min(m, _pairwise_gap(verts_list[i], verts_list[j]))
    return m


def _min_cross(outer_verts, inner_verts_list):
    """Min over inner vertices of the cross product against each outer edge."""
    m = np.inf
    for p in inner_verts_list:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            m = min(m, float(np.cross(p2 - p1, p - p1)))
    return m


def _feasible(verts_list, outer_verts):
    if _min_gap(verts_list) < _EPS:
        return False
    if _min_cross(outer_verts, verts_list) < -_EPS:
        return False
    return True


def _side_from_verts(outer_verts):
    d = outer_verts[1] - outer_verts[0]
    return float(np.linalg.norm(d))


def _honeycomb_start():
    """11 unit hexagons in a compact hexagonal (flower + ring) arrangement."""
    dx = 2.0
    dy = np.sqrt(3.0)
    return np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
        [dx / 2, dy, 0.0],
        [-dx / 2, dy, 0.0],
        [-dx, 0.0, 0.0],
        [-dx / 2, -dy, 0.0],
        [dx / 2, -dy, 0.0],
        [2 * dx, 0.0, 0.0],
        [1.5 * dx, 1.5 * dy, 0.0],
        [0.0, 2 * dy, 0.0],
        [-1.5 * dx, 1.5 * dy, 0.0],
    ])


def _tighten_outer(inner):
    """Compute the minimal-side regular hexagon (centered at origin, angle 0)
    containing all inner vertices. Returns (center, angle_deg, side)."""
    all_pts = []
    for (x, y, a) in inner:
        all_pts.append(_hex_verts(x, y, 1.0, a))
    all_pts = np.vstack(all_pts)

    best = None
    for ang_deg in range(0, 30, 1):
        a = np.deg2rad(ang_deg)
        nrm = np.array([np.cos(a), np.sin(a)])
        p = all_pts @ nrm
        c = 0.5 * (p.max() + p.min())
        r = 0.5 * (p.max() - p.min())
        side = r / np.cos(np.pi / 6)
        if best is None or side < best[3]:
            best = (c, ang_deg, 0.0, side)
    # refine around best angle
    base_ang = best[1]
    for ang_deg in np.linspace(base_ang - 2, base_ang + 2, 41):
        a = np.deg2rad(ang_deg)
        nrm = np.array([np.cos(a), np.sin(a)])
        p = all_pts @ nrm
        c = 0.5 * (p.max() + p.min())
        r = 0.5 * (p.max() - p.min())
        side = r / np.cos(np.pi / 6)
        if side < best[3]:
            best = (c, float(ang_deg), 0.0, side)
    return best[0], best[1], best[3]


def _build_result(inner, center, ang_deg, side):
    # shift inner so that outer is centered at origin
    shifted = inner.copy()
    shifted[:, 0] = inner[:, 0] - center * np.cos(np.deg2rad(ang_deg))
    shifted[:, 1] = inner[:, 1] - center * np.sin(np.deg2rad(ang_deg))

    # verify edge lengths ~ 1
    for (x, y, a) in shifted:
        v = _hex_verts(x, y, 1.0, a)
        for i in range(6):
            L = np.linalg.norm(v[(i + 1) % 6] - v[i])
            if abs(L - 1.0) > 1e-8:
                raise ValueError("edge length off")

    outer = np.array([0.0, 0.0, ang_deg])
    return shifted, outer, float(side)


def _optimize(start, max_iter, time_budget):
    """SLSQP: minimize side by optimizing inner positions + outer center/angle."""
    inner0 = start.copy()
    c0, a0, s0 = _tighten_outer(inner0)
    x0 = np.concatenate([inner0.ravel(), [c0, a0, s0]])

    def objective(x):
        inner = x[:33].reshape(11, 3)
        c, ang, s = x[33], x[34], x[35]
        # containment: shift inner by -center, check cross vs outer at origin
        shifted = inner.copy()
        ca = np.cos(np.deg2rad(ang))
        sa = np.sin(np.deg2rad(ang))
        shifted[:, 0] = inner[:, 0] - c * ca
        shifted[:, 1] = inner[:, 1] - c * sa
        ov = _hex_verts(0.0, 0.0, s, ang)
        mc = _min_cross(ov, [_hex_verts(*row) for row in shifted])
        if mc < -_EPS:
            return s + 1e6 * (_EPS - mc)
        return s

    def con_sep(x):
        inner = x[:33].reshape(11, 3)
        vl = [_hex_verts(*row) for row in inner]
        m = _min_gap(vl)
        return m - _EPS

    def con_contain(x):
        inner = x[:33].reshape(11, 3)
        c, ang, s = x[33], x[34], x[35]
        ca = np.cos(np.deg2rad(ang))
        sa = np.sin(np.deg2rad(ang))
        shifted = inner.copy()
        shifted[:, 0] = inner[:, 0] - c * ca
        shifted[:, 1] = inner[:, 1] - c * sa
        ov = _hex_verts(0.0, 0.0, s, ang)
        mc = _min_cross(ov, [_hex_verts(*row) for row in shifted])
        return mc + _EPS

    cons = [
        {'type': 'ineq', 'fun': con_sep},
        {'type': 'ineq', 'fun': con_contain},
    ]
    bounds = [(-20, 20)] * 33 + [(-20, 20), (-180, 180), (np.sqrt(11), 60)]

    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': max_iter,
                                              'ftol': 1e-9,
                                              'disp': False})
    return res


def hexagon_packing_11():
    best_side = None
    best_inner = None
    best_center = None
    best_ang = None

    starts = [_honeycomb_start()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        pert = _honeycomb_start()
        pert[:, 0] += rng.normal(0, 0.25, 11)
        pert[:, 1] += rng.normal(0, 0.25, 11)
        starts.append(pert)

    for idx, st in enumerate(starts):
        if time.time() > _DEADLINE - 90:
            break
        try:
            res = _optimize(st, max_iter=200, time_budget=90)
        except Exception:
            continue
        if res.x is None:
            continue
        inner = res.x[:33].reshape(11, 3)
        c, ang = res.x[33], res.x[34]
        # recompute tight outer to ensure minimal side for this inner
        c2, a2, s2 = _tighten_outer(inner)
        side = s2
        # validate
        shifted = inner.copy()
        ca = np.cos(np.deg2rad(a2))
        sa = np.sin(np.deg2rad(a2))
        shifted[:, 0] = inner[:, 0] - c2 * ca
        shifted[:, 1] = inner[:, 1] - c2 * sa
        ov = _hex_verts(0.0, 0.0, side, a2)
        vl = [_hex_verts(*row) for row in shifted]
        if _min_gap(vl) < _EPS:
            continue
        if _min_cross(ov, vl) < -_EPS:
            continue
        if side < np.sqrt(11) - 1e-9:
            continue
        if best_side is None or side < best_side:
            best_side = side
            best_inner = inner
            best_center = c2
            best_ang = a2

    if best_inner is None:
        # fallback to honeycomb with a safe outer
        inner = _honeycomb_start()
        c, a, s = _tighten_outer(inner)
        # ensure a generous outer
        s = max(s, 6.0)
        return _build_result(inner, c, a, s)

    return _build_result(best_inner, best_center, best_ang, best_side)


# EVOLVE-BLOCK-END
