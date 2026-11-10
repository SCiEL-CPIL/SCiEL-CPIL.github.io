# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 600.0
_EPS = 1e-6
_SQ3 = np.sqrt(3.0)


def _hex_verts(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.column_stack([np.cos(th), np.sin(th)])


def _pair_gap(va, vb, na, nb):
    """Max gap over the 12 edge normals (6 of A, 6 of B). >0 => separated."""
    best = -np.inf
    for n in (na, nb):
        pa = va @ n
        pb = vb @ n
        g = max(pb.min() - pa.max(), pa.min() - pb.max())
        if g > best:
            best = g
    return best


def _outer_side(shifted, a_deg):
    """Minimal side of a regular hexagon (orientation a_deg, centered at origin)
    containing the point set `shifted`."""
    n = _face_normals(a_deg)
    dots = shifted @ n.T
    r = dots.max()
    return 2.0 * r / _SQ3


def _best_orientation(shifted, n_ang=121):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0.0, 60.0, n_ang):
        s = _outer_side(shifted, a_deg)
        if s < best_s:
            best_s, best_a = s, a_deg
    return best_s, best_a


def _objective(params):
    """Returns (side, min_gap)."""
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_verts(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap(verts[i], verts[j], norms[i], norms[j])
            if g < min_gap:
                min_gap = g
    allv = verts.reshape(-1, 2)
    cent = allv.mean(axis=0)
    shifted = allv - cent
    side, _ = _best_orientation(shifted, n_ang=25)
    return side, min_gap


def _objective_scalar(params):
    side, min_gap = _objective(params)
    if min_gap < _EPS:
        return 1e6 * (_EPS - min_gap) + side
    return side


def _valid(params):
    """Contract validity. Returns (side, outer) if valid else None."""
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_verts(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    for i in range(11):
        for j in range(i + 1, 11):
            if _pair_gap(verts[i], verts[j], norms[i], norms[j]) <= _EPS:
                return None
    allv = verts.reshape(-1, 2)
    cent = allv.mean(axis=0)
    shifted = allv - cent
    side, a_deg = _best_orientation(shifted, n_ang=121)
    if side < np.sqrt(11.0) - 1e-9:
        return None
    # Exact containment check at the centroid (contract shifts by outer center).
    ov = _hex_verts(0.0, 0.0, side, a_deg)
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        e = p2 - p1
        c = np.cross(e, shifted - p1)
        if c.min() < -1e-6:
            return None
    return side, (cent[0], cent[1], a_deg)


def _flower_seed():
    p = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4, 5]:
        ang = np.deg2rad(60.0 * k + 30.0)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _ring_seed():
    p = []
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4]:
        ang = np.deg2rad(60.0 * k)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    for k in [1, 3, 5]:
        ang = np.deg2rad(60.0 * k)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _band_seed():
    # 3-2-3-2-3 brick-wall band (honeycomb), rows along x.
    h = _SQ3 * 2.0
    rows = [
        (0.0, 3), (h, 2), (2 * h, 3), (3 * h, 2), (4 * h, 3),
    ]
    p = []
    for y, cnt in rows:
        for i in range(cnt):
            p.append([(i - (cnt - 1) / 2.0) * 2.0, y, 0.0])
    return np.array(p).flatten()


def hexagon_packing_11():
    seeds = [_flower_seed(), _ring_seed(), _band_seed()]

    best = None          # (side, inner(11,3), outer(3,))
    best_side = np.inf

    def consider(params):
        nonlocal best, best_side
        v = _valid(params)
        if v is None:
            return
        side, outer = v
        if side < best_side:
            best_side = side
            inner = np.empty((11, 3))
            for i in range(11):
                inner[i] = [params[3 * i] - outer[0],
                            params[3 * i + 1] - outer[1],
                            params[3 * i + 2]]
            best = (side, inner, np.array(outer))

    for seed in seeds:
        if time.time() > _DEADLINE - 30.0:
            break
        res = minimize(_objective_scalar, seed, method='Nelder-Mead',
                       options={'maxiter': 2500, 'xatol': 1e-7,
                                'fatol': 1e-7, 'adaptive': True})
        consider(res.x)
        if time.time() > _DEADLINE - 60.0:
            break

    if best is None:
        # Fallback: a guaranteed-valid (large) grid packing.
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        return inner, np.array([0.0, 0.0, 0.0]), 8.0

    side, inner, outer = best
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
