"""Optimized packing: lattice of 12 unit hexagons + SLSQP polish + fine angle search."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# Unit regular hexagon vertices (flat-top, side 1), centered at origin.
# Flat-top: adjacent vertices differ by 60 deg; vertex0 at angle 0.
_UH = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                for a in range(0, 360, 60)])


def _rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _min_dist(pts, center):
    """Min distance from center to polygon edge (inradius when center inside)."""
    n = len(pts)
    best = np.inf
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        e = p2 - p1
        nrm = np.array([-e[1], e[0]])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-15)
        # signed distance from center to the edge line
        d = np.dot(center - p1, nrm)
        best = min(best, d)
    return best


def _inner_vertices(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    return _UH @ np.array([[math.cos(a), -math.sin(a)],
                           [math.sin(a), math.cos(a)]]) + np.array([cx, cy])


def _check(inner, ocx, ocy, side, oang_deg):
    """Return True if all inner hexagons are disjoint and inside outer."""
    # outer vertices
    ov = _rot(_UH, math.radians(oang_deg)) * side + np.array([ocx, ocy])
    # inner inside outer
    for (ix, iy, ia) in inner:
        d = _min_dist(ov, np.array([ix, iy]))
        if d < -1e-7:
            return False
    # pairwise disjoint
    n = len(inner)
    for i in range(n):
        vi = _inner_vertices(inner[i][0], inner[i][1], inner[i][2])
        for j in range(i + 1, n):
            vj = _inner_vertices(inner[j][0], inner[j][1], inner[j][2])
            # check each edge of i vs vertices of j and vice versa
            if _separate(vi, vj) and _separate(vj, vi):
                continue
            # if not separated, they intersect
            return False
    return True


def _separate(polyA, polyB):
    """True if there exists a separating axis (SAT)."""
    for poly in (polyA, polyB):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            nrm = np.array([-e[1], e[0]])
            nrm = nrm / (np.linalg.norm(nrm) + 1e-15)
            projA = np.dot(polyA, nrm)
            projB = np.dot(polyB, nrm)
            if projA.max() < projB.min() - 1e-9 or projB.max() < projA.min() - 1e-9:
                return True
    return False


def _lattice_layout(rows, cols, ang_deg):
    """Hexagonal close-packed lattice of unit hexagons."""
    # flat-top hex: width = 2, height = sqrt(3)
    # horizontal spacing for flat-top adjacent = 1.5 (3/2)
    # vertical spacing = sqrt(3)
    dx = 1.5
    dy = math.sqrt(3)
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += dx / 2.0
            pts.append([x, y, ang_deg])
    return pts


def _score_layout(inner, oang_deg):
    """Compute minimal outer side length for given inner layout and outer angle.
    Returns (side, valid) by solving for minimal enclosing hexagon."""
    # We need to find minimal side s and center such that all inner are inside.
    # Use SLSQP to minimize s over (cx, cy, s).
    oang = math.radians(oang_deg)
    ov_base = _rot(_UH, oang)  # unit outer hex vertices

    def _viol(x):
        cx, cy, s = x
        ov = ov_base * s + np.array([cx, cy])
        v = 0.0
        for (ix, iy, ia) in inner:
            d = _min_dist(ov, np.array([ix, iy]))
            if d < 0:
                v += (-d) ** 2
        return v

    def _obj(x):
        return x[2]

    # initial guess
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx0 = (max(xs) + min(xs)) / 2.0
    cy0 = (max(ys) + min(ys)) / 2.0
    s0 = max(max(xs) - min(xs), max(ys) - min(ys)) + 4.0

    cons = []
    for (ix, iy, ia) in inner:
        def _mk(ix, iy):
            def c(x, ix=ix, iy=iy):
                cx, cy, s = x
                ov = ov_base * s + np.array([cx, cy])
                d = _min_dist(ov, np.array([ix, iy]))
                return d
            return c
        cons.append({'type': 'ineq', 'fun': _mk(ix, iy)})

    res = minimize(_obj, [cx0, cy0, s0], method='SLSQP',
                   constraints=cons, bounds=[(None, None), (None, None), (2.0, None)],
                   options={'maxiter': 200, 'ftol': 1e-10})
    if res.success or res.fun < s0:
        return res.fun, True
    return s0, False


def construct():
    t0 = time.time()
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_oang = 0.0

    # Candidate lattice topologies (rows x cols) and inner hex rotation
    topologies = [
        (3, 4), (4, 3), (2, 6), (6, 2), (1, 12), (12, 1), (2, 3), (3, 2),
    ]
    # Filter to those with exactly N cells
    topologies = [t for t in topologies if t[0] * t[1] == N]
    if not topologies:
        topologies = [(3, 4)]

    inner_angles = [0.0, 30.0, 60.0, 90.0, 15.0, 45.0, 75.0, 120.0]
    outer_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0,
                    10.0, 20.0, 25.0, 35.0, 40.0, 50.0, 55.0, 65.0, 70.0, 80.0]

    # Coarse search
    for (rows, cols) in topologies:
        for ia in inner_angles:
            inner = _lattice_layout(rows, cols, ia)
            if len(inner) != N:
                continue
            for oa in outer_angles:
                if time.time() > _DEADLINE:
                    break
                side, ok = _score_layout(inner, oa)
                if ok and side < best_side:
                    best_side = side
                    best_inner = [list(p) for p in inner]
                    # recompute center from best
                    best_oang = oa
            if time.time() > _DEADLINE:
                break
        if time.time() > _DEADLINE:
            break

    if best_inner is None:
        # fallback: trivial row
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Determine center from the best layout via a final SLSQP
    oang = math.radians(best_oang)
    ov_base = _rot(_UH, oang)

    def _viol(x):
        cx, cy, s = x
        ov = ov_base * s + np.array([cx, cy])
        v = 0.0
        for (ix, iy, ia) in best_inner:
            d = _min_dist(ov, np.array([ix, iy]))
            if d < 0:
                v += (-d) ** 2
        return v

    def _obj(x):
        return x[2]

    xs = [p[0] for p in best_inner]
    ys = [p[1] for p in best_inner]
    cx0 = (max(xs) + min(xs)) / 2.0
    cy0 = (max(ys) + min(ys)) / 2.0

    cons = []
    for (ix, iy, ia) in best_inner:
        def _mk(ix, iy):
            def c(x, ix=ix, iy=iy):
                cx, cy, s = x
                ov = ov_base * s + np.array([cx, cy])
                d = _min_dist(ov, np.array([ix, iy]))
                return d
            return c
        cons.append({'type': 'ineq', 'fun': _mk(ix, iy)})

    res = minimize(_obj, [cx0, cy0, best_side], method='SLSQP',
                   constraints=cons, bounds=[(None, None), (None, None), (2.0, None)],
                   options={'maxiter': 500, 'ftol': 1e-12})
    if res.success or res.fun < best_side:
        best_side = res.fun
        best_center = (float(res.x[0]), float(res.x[1]))
    else:
        best_center = (cx0, cy0)

    return best_inner, best_center, float(best_side), float(best_oang)
