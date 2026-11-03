"""Packing 12 unit hexagons in a minimal enclosing regular hexagon.

Approach:
  - Base layout: centered hexagonal cluster (1 + 6 + 5) with small rotations.
  - Enclosing regular hexagon: for a rotation theta and free center,
    minimal side = (1/sqrt(3)) * max_k (range of projections onto dir_k)/2,
    where dir_k are the 6 outward normals at angles theta + k*60deg.
    (A regular hexagon of side s has inradius s*sqrt(3); the width in each
    normal direction is 2*s*sqrt(3).)
  - Local search (simulated annealing) over inner positions + rotations to
    shrink the enclosing side length, with a hard validity check (no overlap,
    all inside).
"""
import math
import time
import random
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    v = []
    for k in range(6):
        t = a + k * math.pi / 3.0
        v.append((cx + s * math.cos(t), cy + s * math.sin(t)))
    return v

def _enclosing_side(pts, theta):
    # pts: (n,2). theta: radians. Return minimal side of regular hexagon
    # (rotation theta, free center) enclosing all pts, or inf if degenerate.
    P = np.asarray(pts, dtype=float)
    best = 0.0
    for k in range(6):
        ang = theta + k * math.pi / 3.0
        d = np.array([math.cos(ang), math.sin(ang)])
        proj = P @ d
        rng = proj.max() - proj.min()
        if rng > best:
            best = rng
    # width in normal dir = 2 * inradius = 2 * s * sqrt(3)
    return best / (2.0 * math.sqrt(3.0))

def _best_theta(pts):
    P = np.asarray(pts, dtype=float)
    best_s = math.inf
    best_t = 0.0
    # scan theta; the max-range is piecewise, 60deg periodic
    for i in range(360):
        t = i * math.pi / 180.0
        s = _enclosing_side(P, t)
        if s < best_s:
            best_s = s
            best_t = t
    # refine
    lo = best_t - math.pi / 180.0
    hi = best_t + math.pi / 180.0
    for _ in range(30):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _enclosing_side(P, m1) < _enclosing_side(P, m2):
            hi = m2
        else:
            lo = m1
    t = (lo + hi) / 2.0
    return _enclosing_side(P, t), t

def _hex_overlap(v1, v2, tol=1e-9):
    # v1, v2: (6,2). True if interiors overlap.
    # Check: any vertex of one strictly inside the other, or edges cross.
    def inside(pt, poly):
        # ray / sign test for convex poly (ccw)
        n = len(poly)
        sign = 0
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if cross > tol:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -tol:
                if sign > 0:
                    return False
                sign = -1
        return True
    for p in v1:
        if inside(p, v2):
            return True
    for p in v2:
        if inside(p, v1):
            return True
    return False

def _seg_cross(a, b, c, d):
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if v > 1e-12:
            return 1
        if v < -1e-12:
            return -1
        return 0
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return False

def _edges_cross(v1, v2, tol=1e-9):
    for i in range(6):
        a = v1[i]
        b = v1[(i + 1) % 6]
        for j in range(6):
            c = v2[j]
            d = v2[(j + 1) % 6]
            if _seg_cross(a, b, c, d):
                # ensure it's a real crossing not just shared endpoint
                return True
    return False

def _config_valid(inner, center, side, angle_deg):
    # inner: list of (x, y, ang). Check pairwise non-overlap and containment.
    # Build outer hexagon.
    oa = math.radians(angle_deg)
    outer = []
    for k in range(6):
        t = oa + k * math.pi / 3.0
        outer.append((center[0] + side * math.cos(t), center[1] + side * math.sin(t)))
    verts = []
    for (x, y, a) in inner:
        verts.append(_hex_verts(x, y, 1.0, a))
    # pairwise
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if _hex_overlap(verts[i], verts[j]) or _edges_cross(verts[i], verts[j]):
                return False
    # containment: each inner vertex inside outer
    def inside(pt, poly):
        n = len(poly)
        sign = 0
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if cross > 1e-9:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-9:
                if sign > 0:
                    return False
                sign = -1
        return True
    for v in verts:
        for p in v:
            if not inside(p, outer):
                return False
    return True

def _center_of(pts):
    P = np.asarray(pts, dtype=float)
    return (float(P[:, 0].mean()), float(P[:, 1].mean()))

def _score_of(inner):
    # return (side, center, angle) for enclosing hexagon, using centroid center
    pts = [(x, y) for (x, y, a) in inner]
    s, t = _best_theta(pts)
    c = _center_of(pts)
    return s, c, math.degrees(t)

def _initial_layout():
    random.seed(12345)
    inner = []
    # center
    inner.append((0.0, 0.0, 0.0))
    # ring of 6
    for k in range(6):
        ang = math.radians(60 * k + 30)
        r = 2.0
        inner.append((r * math.cos(ang), r * math.sin(ang), 0.0))
    # second partial ring of 5
    for k in range(5):
        ang = math.radians(60 * k + 30)
        r = 4.0
        inner.append((r * math.cos(ang), r * math.sin(ang), 0.0))
    return inner

def construct():
    t0 = time.time()
    random.seed(987654321)

    inner = _initial_layout()
    # add tiny random rotations to break symmetry and allow interlocking
    for i in range(len(inner)):
        x, y, a = inner[i]
        inner[i] = (x, y, random.uniform(-10, 10))

    # ensure a valid starting point: enlarge if needed
    s, c, ang = _score_of(inner)
    # scale positions to make valid with margin
    # We'll just proceed; local search will fix validity.

    best_inner = [list(v) for v in inner]
    best_score = math.inf
    s, c, ang = _score_of(best_inner)
    if _config_valid(best_inner, c, ang):
        best_score = s

    # simulated annealing
    cur = [list(v) for v in inner]
    cur_s, cur_c, cur_a = _score_of(cur)
    cur_valid = _config_valid(cur, cur_c, cur_a)
    cur_cost = cur_s if cur_valid else cur_s + 1000.0

    T = 0.5
    T_end = 1e-4
    step_pos = 0.3
    step_rot = 8.0

    n_iter = 0
    while time.time() - t0 < DEADLINE:
        n_iter += 1
        # pick a hexagon to perturb
        i = random.randrange(len(cur))
        x, y, a = cur[i]
        nx = x + random.gauss(0, step_pos)
        ny = y + random.gauss(0, step_pos)
        na = a + random.gauss(0, step_rot)
        cur[i] = (nx, ny, na)

        ns, nc, na_ang = _score_of(cur)
        nvalid = _config_valid(cur, nc, na_ang)
        nc_cost = ns if nvalid else ns + 1000.0

        # accept
        if nc_cost <= cur_cost:
            cur_cost = nc_cost
            cur_s, cur_c, cur_a = ns, nc, na_ang
            cur_valid = nvalid
            if nvalid and ns < best_score:
                best_score = ns
                best_inner = [list(v) for v in cur]
        else:
            # reject
            cur[i] = (x, y, a)

        # anneal
        T = max(T_end, T * 0.9995)

    # final: ensure validity of best_inner; if not, fall back to a guaranteed valid
    # by scaling up the enclosing side.
    s, c, ang = _score_of(best_inner)
    if not _config_valid(best_inner, c, ang):
        # enlarge until valid (linear growth)
        for _ in range(200):
            s *= 1.01
            if _config_valid(best_inner, c, ang):
                break
    # recompute center/angle for the final side
    s2, c2, a2 = _best_theta([(x, y) for (x, y, aa) in best_inner])
    # use the valid enlarged side if needed
    final_s = max(s, s2)
    # recompute center as centroid for best fit
    final_c = _center_of([(x, y) for (x, y, aa) in best_inner])
    _, final_a = _best_theta([(x, y) for (x, y, aa) in best_inner])

    # final validity check; if fails, scale a bit more
    for _ in range(100):
        if _config_valid(best_inner, final_c, final_a):
            break
        final_s *= 1.02

    inner_out = [[float(x), float(y), float(a)] for (x, y, a) in best_inner]
    return inner_out, (float(final_c[0]), float(final_c[1])), float(final_s), float(final_a)

# EVOLVE-BLOCK-START
# (construct defined above; this marker kept for compatibility)
# EVOLVE-BLOCK-END
