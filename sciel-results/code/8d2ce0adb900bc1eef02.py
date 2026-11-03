"""Hexagon packing: 11 unit hexagons in a hexagonal lattice pattern.
Minimize the enclosing regular-hexagon side length.
Family: fixed 11-hex lattice configuration + optimize outer center/rotation.
Improvements: vectorized coarse scan, smooth log-sum-exp epigraph objective,
gradient polish (L-BFGS-B) + exact Nelder-Mead + exact final polish."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
# Unit hexagon (side 1) circumradius = 1, apothem = SQRT3/2
APOTHEM = SQRT3 / 2.0
# Normal (apothem) directions of a regular hexagon, degrees
NORM_DEG = np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0])


def _hex_vertices(cx, cy, side, angle_deg):
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2.0
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        if lx * math.cos(a_norm) + ly * math.sin(a_norm) > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes, eps=1e-9):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _all_in_outer(inner_hexes, outer_hex):
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            return False
    return True


def _min_outer_side_exact(hexes, ox, oy, oangle):
    """Exact min side length of outer hexagon (center ox,oy; rotation oangle)
    containing all hexes in `hexes` (list of (cx,cy,s,a))."""
    if not hexes:
        return 0.0
    all_verts = []
    for cx, cy, s, ang in hexes:
        all_verts.extend(_hex_vertices(cx, cy, s, ang))
    if not all_verts:
        return 0.0
    verts_arr = np.array(all_verts)
    verts_arr[:, 0] -= ox
    verts_arr[:, 1] -= oy
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    lx = verts_arr[:, 0] * cos_a - verts_arr[:, 1] * sin_a
    ly = verts_arr[:, 0] * sin_a + verts_arr[:, 1] * cos_a
    nx = np.cos(NORM_DEG)
    ny = np.sin(NORM_DEG)
    projs = lx[:, None] * nx[None, :] + ly[:, None] * ny[None, :]
    max_apothem = float(np.max(projs))
    return max_apothem * 2.0 / SQRT3


def _smooth_max(tau):
    """Smooth approximation of max(z) via log-sum-exp, with local scaling
    for numerical stability. tau controls smoothness (smaller = closer to max)."""
    def f(z):
        z = np.asarray(z, dtype=float)
        m = z.max()
        # scale so that exp is well-conditioned: exp((z-m)/tau) in (0,1]
        return m + tau * math.log(np.sum(np.exp((z - m) / tau)))
    return f


def _build_positions():
    """Return list of 11 (x,y) positions for the lattice packing (center + ring1 + 4 ring2)."""
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    return pos_axial, ring2, axial_to_xy


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    pos_axial, ring2, axial_to_xy = _build_positions()
    ring2_idx = list(range(len(ring2)))
    all_combos = list(itertools.combinations(ring2_idx, 4))

    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # ---- Phase 1: coarse global scan (vectorized over outer orientations) ----
    base_xy = [axial_to_xy(q, r) for q, r in pos_axial]
    r2_xy = [axial_to_xy(q, r) for q, r in ring2]

    for combo in all_combos:
        if elapsed() > deadline - 70.0:  # reserve time for optimization phases
            break
        full_xy = base_xy + [r2_xy[i] for i in combo]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]

        # All vertices of the 11 hexagons (66 points)
        all_verts = []
        for cx, cy, s, ang in inner_hexes:
            all_verts.extend(_hex_vertices(cx, cy, s, ang))
        V = np.array(all_verts)  # (66, 2)

        # Centroid (outer center initial guess)
        cx_mean = float(np.mean([p[0] for p in full_xy]))
        cy_mean = float(np.mean([p[1] for p in full_xy]))

        # Vectorized over a grid of outer orientations
        oang = np.arange(0.0, 180.0, 3.0)
        a = -np.deg2rad(oang)
        ca = np.cos(a)  # (M,)
        sa = np.sin(a)  # (M,)
        # Local coords relative to centroid
        lx = (V[:, 0] - cx_mean) * ca[None, :] - (V[:, 1] - cy_mean) * sa[None, :]
        ly = (V[:, 0] - cx_mean) * sa[None, :] + (V[:, 1] - cy_mean) * ca[None, :]
        nx = np.cos(NORM_DEG)  # (6,)
        ny = np.sin(NORM_DEG)
        # projs: (66, M, 6)
        projs = lx[:, :, None] * nx[None, None, :] + ly[:, :, None] * ny[None, None, :]
        max_apothem = np.max(projs, axis=0)  # (M,)
        sides = max_apothem * 2.0 / SQRT3
        k = int(np.argmin(sides))
        if sides[k] < best_side:
            best_side = float(sides[k])
            best_positions = full_xy
            best_center = (cx_mean, cy_mean)
            best_angle = float(oang[k])

    if best_positions is None:
        best_positions = base_xy
        best_side = _min_outer_side_exact([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    inner_hexes = [(x, y, 1.0, 1.0, 0.0) for x, y in best_positions]
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # ---- Phase 2: smooth gradient polish (L-BFGS-B on log-sum-exp epigraph) ----
    def smooth_obj(params, tau):
        cx, cy, ang = params
        all_verts = []
        for hcx, hcy, s, hang in inner_hexes:
            all_verts.extend(_hex_vertices(hcx, hcy, s, hang))
        V = np.array(all_verts)
        V[:, 0] -= cx
        V[:, 1] -= cy
        a = -math.radians(ang)
        ca = math.cos(a)
        sa = math.sin(a)
        lx = V[:, 0] * ca - V[:, 1] * sa
        ly = V[:, 0] * sa + V[:, 1] * ca
        nx = np.cos(NORM_DEG)
        ny = np.sin(NORM_DEG)
        projs = lx[:, None] * nx[None, :] + ly[:, None] * ny[None, :]
        max_apothem = float(_smooth_max(tau)(projs))
        return max_apothem * 2.0 / SQRT3

    def exact_obj(params):
        return _min_outer_side_exact(inner_hexes, params[0], params[1], params[2])

    # Anisotropic initial step: center moves more than angle
    step = [0.5, 0.5, 0.1]

    # Nelder-Mead on the exact objective from the coarse best point
    x0 = [best_center[0], best_center[1], best_angle]
    try:
        res = minimize(exact_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                'adaptive': True, 'initial_simplex': step})
        if res.fun < best_side:
            best_side = float(res.fun)
            best_center = (float(res.x[0]), float(res.x[1]))
            best_angle = float(res.x[2])
    except Exception:
        pass

    # L-BFGS-B on the smooth objective (gradient-based, handles the kink better)
    if elapsed() < deadline - 20.0:
        for tau in [0.05, 0.02, 0.01]:
            if elapsed() > deadline - 15.0:
                break
            try:
                res = minimize(lambda p: smooth_obj(p, tau), x0, method='L-BFGS-B',
                               bounds=[(-6, 6), (-6, 6), (-90, 90)],
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                # Evaluate exact objective at the smooth optimum
                ex = exact_obj(res.x)
                if ex < best_side:
                    best_side = float(ex)
                    best_center = (float(res.x[0]), float(res.x[1]))
                    best_angle = float(res.x[2])
                    x0 = list(res.x)
            except Exception:
                pass

    # ---- Phase 3: exact high-precision polish (Nelder-Mead, tight tolerances) ----
    if elapsed() < deadline - 10.0:
        start_points = [
            [best_center[0], best_center[1], best_angle],
            [best_center[0], best_center[1], best_angle + 1.0],
            [best_center[0], best_center[1], best_angle - 1.0],
            [best_center[0] + 0.05, best_center[1], best_angle],
            [best_center[0] - 0.05, best_center[1], best_angle],
            [best_center[0], best_center[1] + 0.05, best_angle],
            [best_center[0], best_center[1] - 0.05, best_angle],
        ]
        for x0p in start_points:
            if elapsed() > deadline - 8.0:
                break
            try:
                res = minimize(exact_obj, x0p, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-14,
                                        'adaptive': True, 'initial_simplex': step})
                if res.fun < best_side:
                    best_side = float(res.fun)
                    best_center = (float(res.x[0]), float(res.x[1]))
                    best_angle = float(res.x[2])
            except Exception:
                pass

    # ---- Phase 4: exact final polish (very tight) ----
    if elapsed() < deadline - 5.0:
        try:
            res = minimize(exact_obj, [best_center[0], best_center[1], best_angle],
                           method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-13, 'fatol': 1e-15,
                                    'adaptive': True, 'initial_simplex': [0.1, 0.1, 0.02]})
            if res.fun < best_side:
                best_side = float(res.fun)
                best_center = (float(res.x[0]), float(res.x[1]))
                best_angle = float(res.x[2])
        except Exception:
            pass

    # Rebuild final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness; add a tiny gap if needed
    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.01, 1.05]:
            scaled = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled):
                inner_hexes = scaled
                cx_mean = sum(h[0] for h in inner_hexes) / N
                cy_mean = sum(h[1] for h in inner_hexes) / N
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                best_side = _min_outer_side_exact(inner_hexes, cx_mean, cy_mean, 0.0)
                break

    # Verify all inner hexagons inside outer; if not, binary search for valid side
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    if not _all_in_outer(inner_hexes, outer_hex):
        lo, hi = best_side, best_side * 1.2
        for _ in range(60):
            if elapsed() > deadline - 2.0:
                break
            mid = (lo + hi) / 2.0
            if _all_in_outer(inner_hexes, (best_center[0], best_center[1], mid, best_angle)):
                hi = mid
            else:
                lo = mid
        best_side = hi
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (best_center[0], best_center[1]), best_side, best_angle
