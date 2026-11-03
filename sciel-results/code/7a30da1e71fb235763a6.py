"""Packing 12 unit hexagons in a regular hexagon via continuous SLSQP optimization."""
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_DEADLINE_MARGIN = 5.0  # seconds to keep under the 120s wall-clock limit
_START = None
_BEST = None
_BEST_SCORE = None


def _now():
    import time
    return time.time()


def _deadline_ok():
    global _START
    if _START is None:
        return True
    return (_now() - _START) < (120.0 - _DEADLINE_MARGIN)


def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    pts = []
    for k in range(6):
        t = a + math.radians(60.0 * k)
        pts.append((cx + s * math.cos(t), cy + s * math.sin(t)))
    return pts


def _outer_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    pts = []
    for k in range(6):
        t = a + math.radians(60.0 * k)
        pts.append((cx + s * math.cos(t), cy + s * math.sin(t)))
    return pts


def _signed_area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _edge_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if v > 1e-12:
            return 1
        if v < -1e-12:
            return -1
        return 0
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False


def _polygons_overlap_sat(a, b):
    # SAT for two convex polygons
    polys = (a, b)
    for poly in polys:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # edge normal
            ex = x2 - x1
            ey = y2 - y1
            # normal
            nx, ny = -ey, ex
            # project both polygons
            min_a = min(nx * p[0] + ny * p[1] for p in a)
            max_a = max(nx * p[0] + ny * p[1] for p in a)
            min_b = min(nx * p[0] + ny * p[1] for p in b)
            max_b = max(nx * p[0] + ny * p[1] for p in b)
            if max_a < min_b - 1e-9 or max_b < min_a - 1e-9:
                return False
    # If no separating axis found, check for actual intersection (could be touching)
    # Determine if they truly overlap (not just touching) by checking point-in-poly or edge cross
    # Use a robust check: is any vertex of a strictly inside b or vice versa, or edges cross
    for p in a:
        if _point_in_poly(p[0], p[1], b):
            return True
    for p in b:
        if _point_in_poly(p[0], p[1], a):
            return True
    na = len(a)
    nb = len(b)
    for i in range(na):
        for j in range(nb):
            if _edge_intersect(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return True
    return False


def _decode(params):
    # params: for each hexagon: x, y, angle (12*3 = 36)
    xs = params[0::3]
    ys = params[1::3]
    angs = params[2::3]
    return xs, ys, angs


def _objective(params):
    xs, ys, angs = _decode(params)
    outer_s = 1.0  # fixed for objective; we optimize placements then compute minimal outer
    # Actually we want to minimize outer side. Approach: fix outer side S (a variable) and
    # require all inner hexes fit inside and don't overlap. But SLSQP is continuous; we can
    # instead minimize the maximum "required outer side" given placements.
    # Simpler: minimize a proxy = max over inner hexes of (distance needed) plus overlap penalty.
    # We'll compute the minimal outer side that contains all inner hexes (with outer centered at origin,
    # angle 0), then add overlap penalty.
    # Minimal outer side for a set of points contained in a regular hexagon centered at origin (angle 0):
    # A regular hexagon (flat-top? pointy?) with side S and center origin, angle 0: vertices at angles 0,60,...
    # The hexagon is the intersection of 6 half-planes. For a regular hexagon with circumradius R = S,
    # the inradius (apothem) = S * sqrt(3)/2. The hexagon = { p : |dot(p, u_k)| <= apothem for k=0..5 }
    # where u_k are the 6 outward normals (perpendicular to edges). For angle 0 orientation, edges are
    # perpendicular to directions at 30,90,150,210,270,330 degrees.
    apothem_required = 0.0
    # normals at 30+60k
    for k in range(6):
        t = math.radians(30.0 + 60.0 * k)
        ux, uy = math.cos(t), math.sin(t)
        # all points of all inner hexes
        for i in range(N):
            verts = _hex_vertices(xs[i], ys[i], 1.0, angs[i])
            for (vx, vy) in verts:
                d = abs(vx * ux + vy * uy)
                if d > apothem_required:
                    apothem_required = d
    # apothem = S * sqrt(3)/2  =>  S = apothem / (sqrt(3)/2) = 2*apothem/sqrt(3)
    outer_s = 2.0 * apothem_required / math.sqrt(3.0)

    # overlap penalty
    pen = 0.0
    verts_list = [_hex_vertices(xs[i], ys[i], 1.0, angs[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _polygons_overlap_sat(verts_list[i], verts_list[j]):
                pen += 10.0
    return outer_s + pen


def _make_seed_honeycomb():
    # 12 hexagons in a honeycomb cluster around origin
    # hex grid axial coords; side 1 -> neighbor distance = sqrt(3)
    d = math.sqrt(3.0)
    seeds = []
    # ring 0: (0,0)
    seeds.append((0.0, 0.0))
    # ring 1: 6 neighbors
    for k in range(6):
        t = math.radians(60.0 * k)
        seeds.append((d * math.cos(t), d * math.sin(t)))
    # ring 2: pick 5 more to total 12
    # ring 2 positions: axial (2,0),(1,1),(0,2),(-1,1),(-2,0),(-1,-1),(0,-2),(1,-2)
    # choose 5
    ring2 = []
    for k in range(6):
        t = math.radians(60.0 * k)
        ring2.append((2 * d * math.cos(t), 2 * d * math.sin(t)))
    # take first 5
    for p in ring2[:5]:
        seeds.append(p)
    return seeds


def _make_seed_grid():
    d = math.sqrt(3.0)
    seeds = []
    # 3x4 grid offset
    for r in range(3):
        for c in range(4):
            x = c * d
            y = r * 1.5 + (0.75 if c % 2 else 0.0)
            seeds.append((x, y))
    return seeds


def _make_seed_random():
    rng = np.random.default_rng(12345)
    seeds = []
    for _ in range(N):
        x = rng.uniform(-2.0, 2.0)
        y = rng.uniform(-2.0, 2.0)
        a = rng.uniform(0.0, 60.0)
        seeds.append((x, y, a))
    return seeds


def _optimize_from_seed(seed):
    # seed: list of (x,y) or (x,y,a)
    x0 = np.zeros(3 * N)
    for i, s in enumerate(seed):
        x0[3 * i] = s[0]
        x0[3 * i + 1] = s[1]
        x0[3 * i + 2] = s[2] if len(s) > 2 else 0.0
    res = minimize(_objective, x0, method='SLSQP',
                   options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
    return res.x, res.fun


def _final_score(params):
    xs, ys, angs = _decode(params)
    verts_list = [_hex_vertices(xs[i], ys[i], 1.0, angs[i]) for i in range(N)]
    # check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _polygons_overlap_sat(verts_list[i], verts_list[j]):
                return None
    # minimal outer side
    apothem_required = 0.0
    for k in range(6):
        t = math.radians(30.0 + 60.0 * k)
        ux, uy = math.cos(t), math.sin(t)
        for (vx, vy) in verts_list[i]:
            pass
    # recompute properly over all
    apothem_required = 0.0
    for k in range(6):
        t = math.radians(30.0 + 60.0 * k)
        ux, uy = math.cos(t), math.sin(t)
        for i in range(N):
            for (vx, vy) in verts_list[i]:
                d = abs(vx * ux + vy * uy)
                if d > apothem_required:
                    apothem_required = d
    outer_s = 2.0 * apothem_required / math.sqrt(3.0)
    return outer_s


def construct():
    global _START, _BEST, _BEST_SCORE
    _START = _now()

    best_params = None
    best_score = None

    seeds = [
        _make_seed_honeycomb(),
        _make_seed_grid(),
        _make_seed_random(),
    ]

    for seed in seeds:
        if not _deadline_ok():
            break
        try:
            params, fun = _optimize_from_seed(seed)
        except Exception:
            continue
        if not _deadline_ok():
            break
        # refine with a couple of random perturbations
        for _ in range(3):
            if not _deadline_ok():
                break
            rng = np.random.default_rng()
            pert = params + rng.normal(0, 0.1, size=3 * N)
            try:
                params2, fun2 = _optimize_from_seed([ (pert[3*i], pert[3*i+1], pert[3*i+2]) for i in range(N)])
            except Exception:
                continue
            if fun2 < fun:
                params, fun = params2, fun

        score = _final_score(params)
        if score is not None:
            if best_score is None or score < best_score:
                best_score = score
                best_params = params

    if best_params is None:
        # fallback: honeycomb seed only
        seed = _make_seed_honeycomb()
        x0 = np.zeros(3 * N)
        for i, s in enumerate(seed):
            x0[3 * i] = s[0]
            x0[3 * i + 1] = s[1]
            x0[3 * i + 2] = s[2] if len(s) > 2 else 0.0
        best_params = x0
        best_score = _final_score(best_params)
        if best_score is None:
            best_score = 40.0

    xs, ys, angs = _decode(best_params)
    inner = [[float(xs[i]), float(ys[i]), float(angs[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_score), 0.0
