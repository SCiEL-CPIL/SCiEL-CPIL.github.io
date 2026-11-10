# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0
_SAFETY = 30.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _outer_vertices(ox, oy, side, a_deg):
    return _hex_vertices(ox, oy, side, a_deg)


def _min_gap_pair(v1, v2):
    """Minimum separation between two convex hexagons (arrays of 6 (x,y) vertices).
    Returns the minimum over the 12 edge normals of the gap (positive = separated).
    If polygons overlap in projection on every normal, returns a negative value."""
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if v1.ndim == 1:
        v1 = v1.reshape(-1, 2)
    if v2.ndim == 1:
        v2 = v2.reshape(-1, 2)
    
    # Edge normals for each hexagon
    def normals(v):
        e = np.roll(v, -1, axis=0) - v
        # outward normal (right-hand, since vertices are CCW? check)
        n = np.column_stack([e[:, 1], -e[:, 0]])
        # normalize
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
        return n
    n1 = normals(v1)
    n2 = normals(v2)
    best = -np.inf
    for n in np.vstack([n1, n2]):
        p1 = v1 @ n
        p2 = v2 @ n
        # gap = min(p2) - max(p1) if separated along +n, or min(p1)-max(p2)
        gap = min(p2.min() - p1.max(), p1.min() - p2.max())
        if gap > best:
            best = gap
    return best


def _inside_hexagon(pts, ov):
    """Check all pts (N,2) are inside convex hexagon ov (6,2) CCW.
    Returns True if all inside (with tolerance)."""
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (pts[:, 1] - p1[1]) - (p2[1] - p1[1]) * (pts[:, 0] - p1[0])
        if np.any(cross < -1e-6):
            return False
    return True


def _validate(inner, outer, side):
    # inner: (11,3) (x,y,angle_deg); outer: (3,) (x,y,angle_deg)
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    # vertices of each inner hex
    vs = []
    for (x, y, ang) in inner:
        vs.append(_hex_vertices(x, y, 1.0, ang))
    # pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            g = _min_gap_pair(vs[i], vs[j])
            if g <= 1e-6:
                return False
    # containment: shift by outer center
    ox, oy, oang = outer
    ov = _outer_vertices(ox, oy, side, oang)
    # all inner vertices must be inside
    for v in vs:
        if not _inside_hexagon(v, ov):
            return False
    # edge lengths within 1e-8 of 1
    for v in vs:
        e = np.roll(v, -1, axis=0) - v
        L = np.linalg.norm(e, axis=1)
        if np.any(np.abs(L - 1.0) > 1e-8):
            return False
    return True


def _enclosing_side(positions, oang_deg):
    """Given inner hex centers (11,2) and orientation of outer hex, find minimal side
    such that a regular hexagon (centered at some point, orientation oang) contains all
    inner hex vertices. We fix outer center = centroid for simplicity but allow shift.
    Here we compute, for a fixed outer orientation and center, the required side."""
    # We'll optimize center and side. For a given orientation, the minimal enclosing
    # regular hexagon: we can binary search side and check feasibility of center.
    # Simpler: compute for a candidate center the max over vertices of the required
    # "radius" in each of the 6 support directions.
    # For a regular hexagon with orientation a and center c, a point p is inside iff
    # for each of 6 outward normals n_k, dot(p - c, n_k) <= side * apothem_factor.
    # Actually the support function: hexagon = {c + t : |proj on each normal| <= side*sqrt(3)/2 * ... }
    # For regular hexagon side s, the distance from center to each edge (apothem) = s*sqrt(3)/2.
    # The 6 edge normals are at angles a + 30 + 60k (perpendicular to edges).
    # A point p is inside iff for all k: dot(p-c, n_k) <= s*sqrt(3)/2.
    # So required s >= max_k dot(p-c, n_k) / (sqrt(3)/2).
    a = np.deg2rad(oang_deg)
    # edge normals
    nangles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(nangles), np.sin(nangles)])
    apothem_factor = np.sqrt(3.0) / 2.0
    return N, apothem_factor


def _required_side_for_center(positions, N, apothem_factor, center):
    # positions: (M,2) all inner vertices
    d = positions - center
    proj = d @ N.T  # (M,6)
    s = proj.max() / apothem_factor
    return s


def _solve_for_positions(positions, oang_deg):
    """Given inner hex centers, find minimal enclosing hexagon (orientation fixed) by
    optimizing center."""
    a = np.deg2rad(oang_deg)
    nangles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(nangles), np.sin(nangles)])
    apothem_factor = np.sqrt(3.0) / 2.0
    # gather all inner vertices
    allv = []
    for (x, y) in positions:
        allv.append(_hex_vertices(x, y, 1.0, 0.0))
    allv = np.vstack(allv)
    # minimize side over center (2 vars)
    def obj(c):
        return _required_side_for_center(allv, N, apothem_factor, c)
    c0 = np.mean(positions, axis=0)
    res = minimize(obj, c0, method='Nelder-Mead',
                   options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10})
    best_c = res.x
    best_s = _required_side_for_center(allv, N, apothem_factor, best_c)
    return best_c, best_s


def _build_inner(positions, angles):
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(11, 2)
    angles = np.asarray(angles)
    if angles.ndim == 0:
        angles = np.full(11, float(angles))
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = positions[i, 0]
        inner[i, 1] = positions[i, 1]
        inner[i, 2] = angles[i]
    return inner


def _total_side(positions, angles, oang_deg):
    c, s = _solve_for_positions(positions, oang_deg)
    return s


def _initial_positions():
    d = np.sqrt(3.0)
    pos = np.zeros((11, 2))
    # center
    pos[0] = [0.0, 0.0]
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(k * 60.0)
        pos[k + 1] = [d * np.cos(ang), d * np.sin(ang)]
    # 4 extra: place at 2nd ring in 4 alternating directions (0,120,240,300? )
    # Actually to keep symmetry, place at distance 2d in directions 60,180,300,? Let's place at
    # the "pointy" second-ring spots: angles 0,120,240,300 at distance 2d? That may overlap ring.
    # Let's place 4 extra at 2d in directions 90, 210, 330, and one more... need 4.
    # Use directions 0,120,240,300 at 2d -> but 0 direction 2d is beyond neighbor at d in 0.
    extra_angles = [np.deg2rad(a) for a in [0, 120, 240, 300]]
    for k, ang in enumerate(extra_angles):
        pos[7 + k] = [2 * d * np.cos(ang), 2 * d * np.sin(ang)]
    angles = np.zeros(11)
    return pos, angles


def hexagon_packing_11():
    pos, angles = _initial_positions()
    best = None
    best_side = np.inf
    start_time = time.time()

    # Try several outer orientations
    for oang in [0.0, 30.0, 15.0, 45.0, 60.0]:
        if time.time() - start_time > _DEADLINE - _SAFETY:
            break
        c, s = _solve_for_positions(pos, oang)
        # build result
        inner = _build_inner(pos, angles)
        outer = np.array([c[0], c[1], oang])
        if _validate(inner, outer, s):
            if s < best_side:
                best_side = s
                best = (inner.copy(), outer.copy(), float(s))

    # Now optimize positions to reduce side (keep a fixed orientation, say 0)
    oang = 0.0
    a = np.deg2rad(oang)
    nangles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(nangles), np.sin(nangles)])
    apothem_factor = np.sqrt(3.0) / 2.0

    def objective(x):
        p = x.reshape(11, 2)
        allv = []
        for i in range(11):
            allv.append(_hex_vertices(p[i, 0], p[i, 1], 1.0, 0.0))
        allv = np.vstack(allv)
        # also enforce separation via penalty
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                g = _min_gap_pair(allv[i], allv[j])
                if g < 1e-6:
                    pen += (1e-6 - g) ** 2
        c = np.mean(p, axis=0)
        s = _required_side_for_center(allv, N, apothem_factor, c)
        return s + 1e6 * pen

    x0 = pos.flatten()
    # Check time before optimization
    if time.time() - start_time < _DEADLINE - _SAFETY:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-9})
        p = res.x.reshape(11, 2)
    else:
        p = pos
    # recompute enclosing
    if time.time() - start_time < _DEADLINE - _SAFETY:
        allv = []
        for i in range(11):
            allv.append(_hex_vertices(p[i, 0], p[i, 1], 1.0, 0.0))
        allv = np.vstack(allv)
        c = np.mean(p, axis=0)
        s = _required_side_for_center(allv, N, apothem_factor, c)
        inner = _build_inner(p, angles)
        outer = np.array([c[0], c[1], oang])
        if _validate(inner, outer, s):
            if s < best_side:
                best_side = s
                best = (inner.copy(), outer.copy(), float(s))

    if best is None:
        # fallback to a known-valid loose packing
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
