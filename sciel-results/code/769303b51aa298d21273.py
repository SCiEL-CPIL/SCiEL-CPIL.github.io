# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0
_SAFETY = 40.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _normals(v):
    e = np.roll(v, -1, axis=0) - v
    n = np.column_stack([e[:, 1], -e[:, 0]])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    return n


def _min_gap_pair(v1, v2):
    n1 = _normals(v1)
    n2 = _normals(v2)
    best = -np.inf
    for n in np.vstack([n1, n2]):
        p1 = v1 @ n
        p2 = v2 @ n
        gap = min(p2.min() - p1.max(), p1.min() - p2.max())
        if gap > best:
            best = gap
    return best


def _inside_hexagon(pts, ov):
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (pts[:, 1] - p1[1]) - (p2[1] - p1[1]) * (pts[:, 0] - p1[0])
        if np.any(cross < -1e-6):
            return False
    return True


def _validate(inner, outer, side):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    vs = []
    for (x, y, ang) in inner:
        vs.append(_hex_vertices(x, y, 1.0, ang))
    for i in range(11):
        for j in range(i + 1, 11):
            g = _min_gap_pair(vs[i], vs[j])
            if g <= 1e-6:
                return False
    ox, oy, oang = outer
    ov = _hex_vertices(ox, oy, side, oang)
    for v in vs:
        if not _inside_hexagon(v, ov):
            return False
    for v in vs:
        e = np.roll(v, -1, axis=0) - v
        L = np.linalg.norm(e, axis=1)
        if np.any(np.abs(L - 1.0) > 1e-8):
            return False
    return True


def _N_for(oang_deg):
    a = np.deg2rad(oang_deg)
    nangles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(nangles), np.sin(nangles)])
    return N, np.sqrt(3.0) / 2.0


def _all_vertices(positions, angles):
    out = []
    for i in range(11):
        out.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    return out


def _side_for_center(allv, N, af, center):
    d = allv - center
    proj = d @ N.T
    return proj.max() / af


def _build(inner_pos, angles, center, oang, side):
    inner = np.zeros((11, 3))
    inner[:, 0] = inner_pos[:, 0]
    inner[:, 1] = inner_pos[:, 1]
    inner[:, 2] = angles
    outer = np.array([center[0], center[1], oang])
    return inner, outer, float(side)


def _initial():
    d = np.sqrt(3.0)
    pos = np.zeros((11, 2))
    pos[0] = [0.0, 0.0]
    for k in range(6):
        ang = np.deg2rad(k * 60.0)
        pos[k + 1] = [d * np.cos(ang), d * np.sin(ang)]
    for k, deg in enumerate([30, 150, 270, 330]):
        ang = np.deg2rad(deg)
        pos[7 + k] = [2 * d * np.cos(ang), 2 * d * np.sin(ang)]
    angles = np.zeros(11)
    return pos, angles


def hexagon_packing_11():
    pos, angles = _initial()
    best = None
    best_side = np.inf

    def consider(positions, angles, oang):
        nonlocal best, best_side
        N, af = _N_for(oang)
        allv = _all_vertices(positions, angles)
        # optimize center
        def objc(c):
            return _side_for_center(allv, N, af, c)
        c0 = np.mean(positions, axis=0)
        res = minimize(objc, c0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
        c = res.x
        s = _side_for_center(allv, N, af, c)
        inner, outer, side = _build(positions, angles, c, oang, s)
        if _validate(inner, outer, side):
            if side < best_side:
                best_side = side
                best = (inner.copy(), outer.copy(), side)

    for oang in [0.0, 30.0, 15.0, 45.0]:
        if time.time() - _START > _DEADLINE - _SAFETY:
            break
        consider(pos, angles, oang)

    # Optimize positions + center jointly for a fixed orientation
    oang = 0.0
    N, af = _N_for(oang)

    def objective(x):
        p = x[:22].reshape(11, 2)
        c = x[22:24]
        allv = _all_vertices(p, angles)
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                g = _min_gap_pair(allv[i], allv[j])
                if g < 1e-6:
                    pen += (1e-6 - g) ** 2
        s = _side_for_center(allv, N, af, c)
        return s + 1e6 * pen

    x0 = np.concatenate([pos.flatten(), np.mean(pos, axis=0)])
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-9})
    p = res.x[:22].reshape(11, 2)
    c = res.x[22:24]
    allv = _all_vertices(p, angles)
    s = _side_for_center(allv, N, af, c)
    inner, outer, side = _build(p, angles, c, oang, s)
    if _validate(inner, outer, side):
        if side < best_side:
            best_side = side
            best = (inner.copy(), outer.copy(), side)

    if best is None:
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ])
        outer = np.array([0, 0, 0])
        return inner, outer, 8.0

    return best


# EVOLVE-BLOCK-END
