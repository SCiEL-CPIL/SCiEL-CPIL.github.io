"""Direct hexagon packing: 11 unit hexagons, hex-lattice family, polished outer fit."""
import math
import time
import itertools
import random

N = 11


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
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
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
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Min side of outer hexagon (center, rotation) containing all hexes.
    Since container is convex, it suffices to check all vertices of all hexes."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Build a large pool of hex-lattice positions (center + rings 1,2,3)
    pool_axial = [(0, 0)]
    for ring in range(1, 4):
        # 6*ring positions
        for k in range(6):
            # generate ring positions
            pass
    # Simpler: generate all axial coords within distance 3
    pool_axial = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # axial distance
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 3:
                pool_axial.append((q, r))

    # Precompute Cartesian for pool
    pool_xy = [axial_to_xy(q, r) for q, r in pool_axial]
    P = len(pool_xy)

    # Precompute vertex lists (side=1, angle=0) for each pool position
    pool_verts = [_hex_vertices(x, y, 1.0, 0.0) for x, y in pool_xy]

    # Initial layout: center + ring1 (7) + first 4 of ring2
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    ring3 = [(3, 0), (3, -1), (3, -2), (3, -3), (2, -3), (1, -3),
             (0, -3), (-1, -3), (-2, -3), (-3, -3), (-3, -2), (-3, -1),
             (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-2, 3), (-1, 3),
             (0, 3), (1, 3), (2, 3), (3, 3)]

    # Index map
    idx = {ax: i for i, ax in enumerate(pool_axial)}
    init_sel = set()
    init_sel.add(idx[(0, 0)])
    for ax in ring1:
        init_sel.add(idx[ax])
    for ax in ring2[:4]:
        init_sel.add(idx[ax])
    assert len(init_sel) == 11

    def polish(xys, ox, oy, oang):
        """Local search to minimize outer side over (ox, oy, oang)."""
        hexes = [(x, y, 1.0, 0.0) for x, y in xys]
        cur_side = _min_outer_side(hexes, ox, oy, oang)
        for step in [0.05, 0.01, 0.002, 0.0004, 0.00008]:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        for da in [-step * 4, step * 4]:
                            s = _min_outer_side(hexes, ox + dx, oy + dy, oang + da)
                            if s < cur_side - 1e-13:
                                ox, oy, oang, cur_side = ox + dx, oy + dy, oang + da, s
                                improved = True
                for dx, dy, da in [(-step, 0, 0), (step, 0, 0),
                                   (0, -step, 0), (0, step, 0),
                                   (0, 0, -step * 4), (0, 0, step * 4)]:
                    s = _min_outer_side(hexes, ox + dx, oy + dy, oang + da)
                    if s < cur_side - 1e-13:
                        ox, oy, oang, cur_side = ox + dx, oy + dy, oang + da, s
                        improved = True
        return ox, oy, oang, cur_side

    # Evaluate a selection: try several base angles, pick best, then polish
    def evaluate_selection(sel):
        xys = [pool_xy[i] for i in sel]
        hexes = [(x, y, 1.0, 0.0) for x, y in xys]
        cxm = sum(x for x, y in xys) / 11.0
        cym = sum(y for x, y in xys) / 11.0
        best = None
        for base_ang in [0.0, 30.0, 15.0, -15.0]:
            # quick estimate at centroid
            s0 = _min_outer_side(hexes, cxm, cym, base_ang)
            if best is None or s0 < best[3]:
                best = (cxm, cym, base_ang, s0)
        if best is None:
            return None
        # polish the best base
        ox, oy, oang, s = polish(xys, best[0], best[1], best[2])
        return (ox, oy, oang, s, xys)

    # Simulated annealing over subsets to find good layouts
    rng = random.Random(12345)
    cur_sel = list(init_sel)
    cur_res = evaluate_selection(cur_sel)
    best_res = cur_res
    T = 0.05
    cool = 0.995

    it = 0
    while time.time() < deadline - 5.0:
        it += 1
        # propose a swap: remove one, add one (keep size 11)
        a = rng.randrange(len(cur_sel))
        b = rng.randrange(P)
        if cur_sel[a] == b:
            continue
        new_sel = cur_sel[:]
        new_sel[a] = b
        new_sel_set = set(new_sel)
        if len(new_sel_set) != 11:
            continue
        new_res = evaluate_selection(new_sel_set)
        if new_res is None:
            continue
        dE = new_res[3] - cur_res[3]
        if dE < 0 or rng.random() < math.exp(-dE / T):
            cur_sel = new_sel
            cur_res = new_res
            if cur_res[3] < best_res[3]:
                best_res = cur_res
        T *= cool
        if T < 1e-4:
            break

    # Also try a few explicit good combos from ring2/ring3 for robustness
    # (best_res already from SA; do final polish on best layout)
    ox, oy, oang, s, xys = best_res

    # Final high-precision polish
    ox, oy, oang, s = polish(xys, ox, oy, oang)

    inner_hexes = [(x, y, 1.0, 0.0) for x, y in xys]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Should not happen for hex lattice; fallback to known-good 7+4 ring2
        fallback_ax = [(0, 0)] + ring1 + ring2[:4]
        xys = [axial_to_xy(q, r) for q, r in fallback_ax]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in xys]
        cxm = sum(x for x, y in xys) / 11.0
        cym = sum(y for x, y in xys) / 11.0
        ox, oy, oang, s = polish(xys, cxm, cym, 0.0)

    # Verify containment; binary search side if needed
    outer_hex = (ox, oy, s, oang)
    valid = all(_inner_in_outer(h, outer_hex) for h in inner_hexes)
    if not valid:
        lo, hi = s, s * 1.1
        for _ in range(60):
            mid = (lo + hi) / 2
            test_outer = (ox, oy, mid, oang)
            if all(_inner_in_outer(h, test_outer) for h in inner_hexes):
                hi = mid
            else:
                lo = mid
        s = hi
        outer_hex = (ox, oy, s, oang)

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (ox, oy), s, oang
