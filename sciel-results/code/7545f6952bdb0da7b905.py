# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 580.0  # safety margin before 630s kill

# Rigid cluster: 12 unit hexagons in a 3-2-3-2-2 hexagonal tiling (axial coords q,r)
_AXIAL = np.array([
    [-1, -2], [0, -2], [1, -2],
    [-1, -1], [0, -1],
    [-1, 0], [0, 0], [1, 0],
    [-1, 1], [0, 1],
    [-1, 2], [0, 2],
], dtype=float)


def _axial_to_cartesian(axial):
    q = axial[:, 0]
    r = axial[:, 1]
    x = np.sqrt(3.0) * (q + 0.5 * r)
    y = 1.5 * r
    return np.column_stack([x, y])


_BASE_CENTERS = _axial_to_cartesian(_AXIAL)
_BASE_CENTER = _BASE_CENTERS.mean(axis=0)
_BASE_CENTERS = _BASE_CENTERS - _BASE_CENTER  # center the cluster at origin


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack([cx + d * np.cos(ang), cy + d * np.sin(ang)])


_BASE_VERTS = _hex_vertices(0, 0, 1.0, 0.0)  # (6,2)
# Outward normals of a regular hexagon with vertices at angles 60i:
# edge i (V_i -> V_{i+1}) outward normal at angle 60i+30
_NORM_ANG = np.deg2rad(60.0) * np.arange(6) + np.deg2rad(30.0)
_NORMALS = np.column_stack([np.cos(_NORM_ANG), np.sin(_NORM_ANG)])  # (6,2)


def _all_vertices(theta):
    """(72,2) vertices of the 12 inner hexagons rotated by theta (deg), cluster at origin."""
    th = np.deg2rad(theta)
    c = np.cos(th)
    s = np.sin(th)
    R = np.array([[c, -s], [s, c]])
    centers = _BASE_CENTERS @ R.T
    verts = centers[:, None, :] + _BASE_VERTS[None, :, :]
    return verts.reshape(-1, 2)


def _required_side(theta, ocx, ocy):
    verts = _all_vertices(theta)
    proj = verts @ _NORMALS.T  # (72,6)
    support = proj.max(axis=0)  # (6,)
    dot_oc = _NORMALS @ np.array([ocx, ocy])  # (6,)
    return float((support - dot_oc).max())


def _objective(x):
    return _required_side(x[0], x[1], x[2])


def _build(theta, ocx, ocy, s):
    th = np.deg2rad(theta)
    c = np.cos(th)
    sn = np.sin(th)
    R = np.array([[c, -sn], [sn, c]])
    centers = _BASE_CENTERS @ R.T
    inner = np.empty((12, 3))
    for i in range(12):
        inner[i, 0] = centers[i, 0]
        inner[i, 1] = centers[i, 1]
        inner[i, 2] = theta
    outer = np.array([ocx, ocy, 0.0])
    return inner, outer, s


def _verify(inner, outer, s):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12.0) - 1e-12:
        return False
    cx, cy, ca = outer
    ov = _hex_vertices(cx, cy, s, ca)
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-9:
                return False
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        e = p2 - p1
        for k in range(12):
            ix, iy, ia = inner[k]
            iv = _hex_vertices(ix, iy, 1.0, ia)
            for v in iv:
                cross = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    for a in range(12):
        ia = inner[a]
        va = _hex_vertices(ia[0], ia[1], 1.0, ia[2])
        for b in range(a + 1, 12):
            ib = inner[b]
            vb = _hex_vertices(ib[0], ib[1], 1.0, ib[2])
            ok = True
            for hexv in (va, vb):
                for i in range(6):
                    p1 = hexv[i]
                    p2 = hexv[(i + 1) % 6]
                    e = p2 - p1
                    n = np.array([-e[1], e[0]])
                    na = va @ n
                    nb = vb @ n
                    if na.max() < nb.min() - 1e-6 or nb.max() < na.min() - 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                return False
    return True


def hexagon_packing_12():
    global _DEADLINE
    best = None
    best_s = float('inf')
    best_x = None

    def consider(x):
        nonlocal best, best_s, best_x
        if time.time() > _DEADLINE - 20:
            return
        s = _required_side(x[0], x[1], x[2])
        if not np.isfinite(s):
            return
        if s < best_s:
            best_s = s
            best_x = np.array(x)
            inner, outer, s2 = _build(x[0], x[1], x[2], s)
            if _verify(inner, outer, s2):
                best = (inner, outer, s2)

    # Coarse scan: theta in [0,60) (period 60), small center offsets
    thetas = np.linspace(0.0, 60.0, 61)
    offs = np.linspace(-1.5, 1.5, 7)
    for t in thetas:
        for ox in offs:
            for oy in offs:
                consider((t, ox, oy))

    # Differential evolution
    if time.time() < _DEADLINE - 80:
        bounds = [(-30.0, 30.0), (-2.0, 2.0), (-2.0, 2.0)]
        try:
            de_res = differential_evolution(
                _objective, bounds, seed=0, maxiter=50, popsize=15,
                tol=1e-9, polish=True, init='sobol'
            )
            consider(de_res.x)
        except Exception:
            pass

    # Nelder-Mead polish from seeds + best
    seeds = []
    for t in [0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5]:
        for ox, oy in [(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)]:
            seeds.append([t, ox, oy])
    if best_x is not None:
        seeds.append(best_x.tolist())
    for x0 in seeds:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-13})
            consider(res.x)
        except Exception:
            pass

    # Final high-precision polish from the best point
    if best_x is not None and time.time() < _DEADLINE - 25:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15})
            consider(res.x)
        except Exception:
            pass

    if best is None:
        # Fallback: valid loose packing (theta=0, centered, large s)
        inner, outer, s = _build(0.0, 0.0, 0.0, 8.0)
        if not _verify(inner, outer, s):
            inner, outer, s = _build(0.0, 0.0, 0.0, 12.0)
        return inner, outer, s
    return best


# EVOLVE-BLOCK-END
