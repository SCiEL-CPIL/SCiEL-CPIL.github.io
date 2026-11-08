"""Packing unit regular hexagons inside a regular hexagon (minimize outer side length)."""
import math
import time

N = 11
DEADLINE = 140.0  # safety margin under 150s kill

def _hex_verts(cx, cy, r, ang):
    """Vertices of a regular hexagon, center (cx,cy), circumradius r, rotation ang (radians)."""
    out = []
    for k in range(6):
        a = ang + math.pi * k / 3.0
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out

def _edge_dist(p, q, a, b):
    """Distance from point p to segment a-b."""
    ax, ay = a; bx, by = b; px, py = p
    dx = bx - ax; dy = by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx; qy = ay + t * dy
    return math.hypot(px - qx, py - qy)

def _hex_min_dist(v1, v2):
    """Minimum distance between two convex hexagons (vertices + edges)."""
    best = 1e30
    for p in v1:
        for q in v2:
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            if d < best:
                best = d
    for i in range(6):
        a = v1[i]; b = v1[(i + 1) % 6]
        for p in v2:
            d = _edge_dist(p, a, b)
            if d < best:
                best = d
        for j in range(6):
            c = v2[j]; d2 = v2[(j + 1) % 6]
            # distance between segments a-b and c-d
            def segseg(p1, p2, p3, p4):
                # min distance between two segments
                best = 1e30
                for pt in (p1, p2):
                    for seg in ((p3, p4),):
                        d = _edge_dist(pt, seg[0], seg[1])
                        if d < best:
                            best = d
                for pt in (p3, p4):
                    d = _edge_dist(pt, p1, p2)
                    if d < best:
                        best = d
                return best
            d = segseg(a, b, c, d2)
            if d < best:
                best = d
    return best

def _min_dist_all(cfg, R=1.0):
    """cfg: list of (x, y, ang). Return (min_pair_dist, n_violation) using vertex-vertex only (fast)."""
    n = len(cfg)
    verts = [_hex_verts(x, y, R, a) for (x, y, a) in cfg]
    best = 1e30
    viol = 0.0
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            d = 1e30
            for p in vi:
                for q in vj:
                    dd = math.hypot(p[0] - q[0], p[1] - q[1])
                    if dd < d:
                        d = dd
            # also point-to-edge for a tighter bound
            for p in vi:
                for k in range(6):
                    dd = _edge_dist(p, vj[k], vj[(k + 1) % 6])
                    if dd < d:
                        d = dd
            if d < best:
                best = d
            if d < 1e-6:
                viol += (1e-6 - d)
    return best, viol

def _support_side(cfg, R=1.0):
    """Min enclosing regular hexagon side length for given inner hexagons (fixed outer orientation)."""
    n = len(cfg)
    verts = [_hex_verts(x, y, R, a) for (x, y, a) in cfg]
    allp = [p for v in verts for p in v]
    # outer hexagon normals at 0,60,...: direction (cos, sin)
    best_side = 1e30
    # center chosen to minimize max projection: for a convex set, the minimal enclosing
    # regular hexagon (fixed orientation) side = (max_i proj - min_i proj) / sqrt(3)
    # where proj is along each of the 3 normal axes (and their opposites).
    for base in range(0, 360, 30):
        axes = []
        for k in range(3):
            a = math.radians(base + k * 60.0)
            axes.append((math.cos(a), math.sin(a)))
        span = 0.0
        for (ux, uy) in axes:
            mn = min(p[0] * ux + p[1] * uy for p in allp)
            mx = max(p[0] * ux + p[1] * uy for p in allp)
            span = max(span, mx - mn)
        side = span / math.sqrt(3.0)
        if side < best_side:
            best_side = side
    return best_side

def _construct_lattice():
    """Compact hexagonal lattice packing (optimal for large N)."""
    # Hexagon lattice: centers spaced so hexagons tile. For unit hexagons (side 1),
    # the tiling has center distance = sqrt(3) along two directions 60 deg apart.
    # Use axial coordinates.
    d = math.sqrt(3.0)  # distance between adjacent hexagon centers in a perfect tiling
    centers = []
    # generate a roughly circular cluster of hex lattice points
    # axial coords (q, r) -> cartesian
    def axial(q, r):
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3.0) / 2.0) * r
        return (x, y)
    # collect lattice points within radius
    maxr = 12
    pts = []
    for q in range(-maxr, maxr + 1):
        for r in range(-maxr, maxr + 1):
            x, y = axial(q, r)
            if math.hypot(x, y) <= maxr * d * 0.9:
                pts.append((q, r, x, y))
    # keep only N closest to origin
    pts.sort(key=lambda t: t[2] * t[2] + t[3] * t[3])
    pts = pts[:N]
    # re-center to centroid
    cx = sum(p[2] for p in pts) / N
    cy = sum(p[3] for p in pts) / N
    inner = [[p[2] - cx, p[3] - cy, 0.0] for p in pts]
    side = _support_side(inner)
    return inner, (0.0, 0.0), side, 0.0

def construct():
    t0 = time.time()
    # Baseline: compact hexagonal lattice (provably near-optimal)
    inner, center, side, ang = _construct_lattice()
    best = (inner, center, side, ang)

    # Optional short local polish via simple coordinate descent (only if time allows)
    if time.time() - t0 < 30.0:
        cur = [list(r) for r in inner]
        cur_side = side
        step = 0.05
        for it in range(400):
            if time.time() - t0 > 120.0:
                break
            improved = False
            for i in range(len(cur)):
                for dx in (-step, step):
                    for dy in (-step, step):
                        cand = [list(r) for r in cur]
                        cand[i][0] += dx
                        cand[i][1] += dy
                        s = _support_side(cand)
                        # ensure no overlaps (min dist > 0)
                        md, _ = _min_dist_all(cand)
                        if md > 1e-6 and s < cur_side - 1e-9:
                            cur = cand
                            cur_side = s
                            improved = True
            if not improved:
                step *= 0.8
                if step < 1e-4:
                    break
        if cur_side < side:
            best = (cur, center, cur_side, ang)
    return best[0], best[1], best[2], best[3]
