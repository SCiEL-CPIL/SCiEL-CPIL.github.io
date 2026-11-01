"""Smooth penalty-based optimization for packing 12 unit hexagons in a regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 150.0
START_TIME = time.time()
EPS = 1e-7
SMOOTH = 0.02


def _deadline_ok():
    return (time.time() - START_TIME) < DEADLINE - 2.0


def _hex_vertices(center, side, angle):
    """Vertices of a regular hexagon (flat-top at angle=0)."""
    cx, cy = center
    a = np.deg2rad(angle)
    ca, sa = np.cos(a), np.sin(a)
    base = np.array([np.cos(np.deg2rad(60 * k)) * side for k in range(6)])
    x = base
    y = np.array([np.sin(np.deg2rad(60 * k)) * side for k in range(6)])
    rx = ca * x - sa * y
    ry = sa * x + ca * y
    return np.column_stack([cx + rx, cy + ry])


def _inside_violation(pts, outer):
    """Max signed distance of points outside the outer hexagon (positive = outside)."""
    oxc, oyc, oside, oang = outer
    ox, oy = _hex_vertices((oxc, oyc), oside, oang)
    n = len(ox)
    maxv = 0.0
    for i in range(n):
        ax, ay = ox[i], oy[i]
        bx, by = ox[(i + 1) % n], oy[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        elen = np.hypot(ex, ey)
        if elen < EPS:
            continue
        # inward normal
        nx, ny = ey / elen, -ex / elen
        # check orientation: ensure inward (dot with (center - edge_mid) > 0)
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        if nx * (oxc - mx) + ny * (oyc - my) < 0:
            nx, ny = -nx, -ny
        d = (pts[:, 0] - ax) * nx + (pts[:, 1] - ay) * ny
        v = np.max(d)
        if v > maxv:
            maxv = v
    return maxv


def _penetration_depth(v1, v2):
    """Max overlap depth between two convex polygons (positive = overlap)."""
    n1, n2 = len(v1), len(v2)
    maxd = 0.0
    for i in range(n1):
        ax, ay = v1[i]
        bx, by = v1[(i + 1) % n1]
        ex, ey = bx - ax, by - ay
        elen = np.hypot(ex, ey)
        if elen < EPS:
            continue
        nx, ny = ey / elen, -ex / elen
        d1 = (v1[:, 0] - ax) * nx + (v1[:, 1] - ay) * ny
        d2 = (v2[:, 0] - ax) * nx + (v2[:, 1] - ay) * ny
        max1 = np.max(d1)
        min2 = np.min(d2)
        if min2 > max1:
            return 0.0  # separated along this axis
        gap = max1 - min2
        if gap > maxd:
            maxd = gap
    for i in range(n2):
        ax, ay = v2[i]
        bx, by = v2[(i + 1) % n2]
        ex, ey = bx - ax, by - ay
        elen = np.hypot(ex, ey)
        if elen < EPS:
            continue
        nx, ny = ey / elen, -ex / ny * 0 + (-ex) / elen
        nx, ny = ey / elen, -ex / elen
        d1 = (v1[:, 0] - ax) * nx + (v1[:, 1] - ay) * ny
        d2 = (v2[:, 0] - ax) * nx + (v2[:, 1] - ay) * ny
        max1 = np.max(d1)
        min2 = np.min(d2)
        if min2 > max1:
            return 0.0
        gap = max1 - min2
        if gap > maxd:
            maxd = gap
    return maxd


def _objective(params):
    """Objective: outer side + penalties for containment and overlap."""
    oside = params[0]
    oxc, oyc = params[1], params[2]
    oang = params[3]
    outer = (oxc, oyc, oside, oang)
    rest = params[4:]
    n = len(rest) // 3

    # build inner vertices
    verts = []
    for i in range(n):
        x, y, ang = rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]
        verts.append(_hex_vertices((x, y), 1.0, ang))

    # containment penalty: all inner vertices inside outer
    contain = 0.0
    for v in verts:
        viol = _inside_violation(v, outer)
        if viol > 0:
            contain += viol

    # overlap penalty
    overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pd = _penetration_depth(verts[i], verts[j])
            if pd > 0:
                overlap += pd

    penalty = 100.0 * (contain + overlap)
    return oside + penalty


def _build_initial(side_guess):
    """Build initial params: outer + a ring/cluster layout of inner hexagons."""
    params = [side_guess, 0.0, 0.0, 0.0]
    # cluster inner hexagons in a small hexagonal packing around origin
    # place them in a grid of hex centers (side 1 -> center spacing ~ 2 for non-overlap)
    spacing = 2.0
    positions = []
    # 12 positions in a compact shape
    layout = [
        (0, 0),
        (spacing, 0), (-spacing, 0),
        (0, spacing * np.sqrt(3) / 1), (0, -spacing * np.sqrt(3) / 1),
        (spacing / 2, spacing * np.sqrt(3) / 2), (-spacing / 2, spacing * np.sqrt(3) / 2),
        (spacing / 2, -spacing * np.sqrt(3) / 2), (-spacing / 2, -spacing * np.sqrt(3) / 2),
        (1.5 * spacing, 0), (-1.5 * spacing, 0),
        (0, 1.5 * spacing * np.sqrt(3) / 1),
    ]
    for (px, py) in layout[:N]:
        params.extend([px, py, 0.0])
    return np.array(params, dtype=float)


def construct():
    best = None
    best_val = float('inf')

    def consider(params):
        nonlocal best, best_val
        if not _deadline_ok():
            return
        val = _objective(params)
        if val < best_val:
            best_val = val
            best = params.copy()

    # Try several initial side guesses and starting layouts
    for side_guess in (5.0, 6.0, 7.0, 8.0):
        if not _deadline_ok():
            break
        params = _build_initial(side_guess)
        consider(params)
        try:
            res = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50},
            )
            if _deadline_ok():
                consider(res.x)
        except Exception:
            pass

    if best is None:
        # fallback to trivial large solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Decode best
    oside = float(best[0])
    oxc, oyc = float(best[1]), float(best[2])
    oang = float(best[3])
    rest = best[4:]
    inner = []
    for i in range(N):
        x, y, ang = float(rest[3 * i]), float(rest[3 * i + 1]), float(rest[3 * i + 2])
        inner.append([x, y, ang])

    return inner, (oxc, oyc), oside, oang
