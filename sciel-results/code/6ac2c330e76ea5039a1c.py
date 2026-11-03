"""Physics + simulated-annealing packing of 12 unit hexagons in a regular hexagon."""
import math
import time
import random

N = 12
DEADLINE = 130.0  # safety margin under 150s kill
_START = time.time()

# Unit regular hexagon vertices (pointy-top), side 1
_HEX_VERTS = [(math.cos(math.radians(60.0 * i + 30.0)),
               math.sin(math.radians(60.0 * i + 30.0))) for i in range(6)]
_HEX_VERTS = [(x, y) for x, y in _HEX_VERTS]


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > DEADLINE


def _transform(pts, cx, cy, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in pts]


def _hex_world(cx, cy, ang):
    return _transform(_HEX_VERTS, cx, cy, ang)


def _outer_world(S, ang):
    # outer hexagon of side S, centered at origin
    return _transform([(S * math.cos(math.radians(60.0 * i + 30.0)),
                        S * math.sin(math.radians(60.0 * i + 30.0))) for i in range(6)],
                      0.0, 0.0, ang)


def _sep_axis_dist(a, b):
    # max separation between convex polygons a, b; negative => overlap depth
    best = -1.0
    for poly in (a, b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            nx = y2 - y1
            ny = x1 - x2
            L = math.hypot(nx, ny)
            if L < 1e-12:
                continue
            nx /= L
            ny /= L
            min_a = min(p[0] * nx + p[1] * ny for p in a)
            max_a = max(p[0] * nx + p[1] * ny for p in a)
            min_b = min(p[0] * nx + p[1] * ny for p in b)
            max_b = max(p[0] * nx + p[1] * ny for p in b)
            if max_a < min_b:
                return math.inf
            if max_b < min_a:
                return math.inf
            sep = min(max_a - min_b, max_b - min_a)
            if sep > best:
                best = sep
    return best


def _min_overlap_depth(a, b):
    # depth = -sep if overlapping else 0
    d = _sep_axis_dist(a, b)
    if d == math.inf:
        return 0.0
    return max(0.0, -d)


def _min_overlap_depth_pair(centers, angs, i, j):
    a = _hex_world(centers[i][0], centers[i][1], angs[i])
    b = _hex_world(centers[j][0], centers[j][1], angs[j])
    return _min_overlap_depth(a, b)


def _max_support(centers, angs, ox, oy):
    # maximum of (p·u) over all vertices of all hexagons, u=(ox,oy)
    best = -1e18
    for i in range(N):
        cx, cy = centers[i]
        ang = angs[i]
        ca, sa = math.cos(ang), math.sin(ang)
        for x, y in _HEX_VERTS:
            wx = cx + ca * x - sa * y
            wy = cy + sa * x + ca * y
            v = wx * ox + wy * oy
            if v > best:
                best = v
    return best


def _bounding_side(centers, angs, outer_ang):
    # minimal side S of outer hexagon (centered at origin, rotation outer_ang)
    # such that all hexagons are inside. For a regular hexagon of side S,
    # the support in direction u is S * |cos(angle_diff)| * 2? Let's derive.
    # Outer hexagon of side S: inradius r = S*sqrt(3)/2, circumradius R = S.
    # Support function h(u) = max over vertices. For a regular hexagon centered at
    # origin with circumradius R=S, the support in direction u is R * max over
    # vertex directions. The max support = R * max_i cos(theta_i - phi) where
    # theta_i are vertex angles. That max is R * cos(min angular distance to a vertex).
    # Simpler: the hexagon is the intersection of 6 half-planes x·n_k <= r (inradius)
    # where n_k are the 6 outward normals. So containment <=> for all k,
    # max_i (p_i · n_k) <= r = S*sqrt(3)/2.
    # Thus S >= max_k (max_i p_i·n_k) * 2/sqrt(3).
    r3 = math.sqrt(3.0)
    maxv = 0.0
    for k in range(6):
        ang_n = math.radians(60.0 * k + 30.0) + outer_ang
        nx = math.cos(ang_n)
        ny = math.sin(ang_n)
        v = _max_support(centers, angs, nx, ny)
        if v > maxv:
            maxv = v
    # also need negative side: min over p·n_k >= -r  =>  -min(p·n_k) <= r
    for k in range(6):
        ang_n = math.radians(60.0 * k + 30.0) + outer_ang
        nx = math.cos(ang_n)
        ny = math.sin(ang_n)
        v = -_max_support(centers, angs, -nx, -ny)
        if v > maxv:
            maxv = v
    return maxv * 2.0 / r3


def _total_overlap(centers, angs):
    tot = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            tot += _min_overlap_depth_pair(centers, angs, i, j)
    return tot


def _physics_step(centers, angs, S, dt, outer_ang):
    # compute forces
    fx = [0.0] * N
    fy = [0.0] * N
    tq = [0.0] * N
    # repulsion between overlapping pairs
    for i in range(N):
        for j in range(i + 1, N):
            a = _hex_world(centers[i][0], centers[i][1], angs[i])
            b = _hex_world(centers[j][0], centers[j][1], angs[j])
            d = _sep_axis_dist(a, b)
            if d == math.inf:
                continue
            if d < 0:
                # overlap; push apart along center line
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                L = math.hypot(dx, dy)
                if L < 1e-9:
                    dx, dy = 1.0, 0.0
                    L = 1.0
                dx /= L
                dy /= L
                f = -d * 2.0
                fx[i] += f * dx
                fy[i] += f * dy
                fx[j] -= f * dx
                fy[j] -= f * dy
                # torque: nudge rotation to reduce overlap
                tq[i] += -d * 0.05 * random.uniform(-1, 1)
                tq[j] += -d * 0.05 * random.uniform(-1, 1)
    # containment: pull toward incenter (origin) scaled by how far outside
    r3 = math.sqrt(3.0)
    for k in range(6):
        ang_n = math.radians(60.0 * k + 30.0) + outer_ang
        nx = math.cos(ang_n)
        ny = math.sin(ang_n)
        # for each hexagon, if its max support in n exceeds r, push back
        r = S * r3 / 2.0
        for i in range(N):
            ca, sa = math.cos(angs[i]), math.sin(angs[i])
            cx, cy = centers[i]
            maxp = -1e18
            for x, y in _HEX_VERTS:
                wx = cx + ca * x - sa * y
                wy = cy + sa * x + ca * y
                v = wx * nx + wy * ny
                if v > maxp:
                    maxp = v
            if maxp > r:
                f = (maxp - r) * 1.5
                fx[i] -= f * nx
                fy[i] -= f * ny
            # opposite side
            minp = 1e18
            for x, y in _HEX_VERTS:
                wx = cx + ca * x - sa * y
                wy = cy + sa * x + ca * y
                v = wx * nx + wy * ny
                if v < minp:
                    minp = v
            if minp < -r:
                f = (-r - minp) * 1.5
                fx[i] += f * nx
                fy[i] += f * ny
    # integrate
    for i in range(N):
        centers[i][0] += fx[i] * dt
        centers[i][1] += fy[i] * dt
        angs[i] = (angs[i] + tq[i] * dt) % (2.0 * math.pi)


def _settle(centers, angs, S, outer_ang, max_iter=200, dt=0.05):
    # symplectic-ish Euler; returns best (centers, angs, overlap, side)
    best_overlap = _total_overlap(centers, angs)
    best_side = _bounding_side(centers, angs, outer_ang)
    best_c = [list(c) for c in centers]
    best_a = list(angs)
    for it in range(max_iter):
        if _deadline_hit():
            break
        _physics_step(centers, angs, S, dt, outer_ang)
        # cool dt
        dt *= 0.995
        ov = _total_overlap(centers, angs)
        side = _bounding_side(centers, angs, outer_ang)
        if ov < best_overlap or (ov <= best_overlap and side < best_side):
            best_overlap = ov
            best_side = side
            best_c = [list(c) for c in centers]
            best_a = list(angs)
        if best_overlap < 1e-6 and best_side <= S + 1e-6:
            break
    return best_c, best_a, best_overlap, best_side


def _sa_move(centers, angs, T):
    # random perturbation
    i = random.randrange(N)
    centers[i][0] += random.gauss(0.0, T * 0.5)
    centers[i][1] += random.gauss(0.0, T * 0.5)
    angs[i] = (angs[i] + random.gauss(0.0, T * 0.3)) % (2.0 * math.pi)


def _search(S, outer_ang, time_budget):
    # initialize with a ring + center arrangement
    centers = [[0.0, 0.0]]
    angs = [0.0]
    # 6 around at radius ~2
    for k in range(6):
        a = math.radians(60.0 * k)
        centers.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
        angs.append(a)
    # 5 more scattered
    for k in range(5):
        a = math.radians(72.0 * k + 30.0)
        centers.append([3.2 * math.cos(a), 3.2 * math.sin(a)])
        angs.append(a)

    T = 1.0
    best_c = [list(c) for c in centers]
    best_a = list(angs)
    best_side = _bounding_side(centers, angs, outer_ang)
    best_ov = _total_overlap(centers, angs)

    it = 0
    while not _deadline_hit() and (time.time() - time_budget) < 0:
        pass
    # main SA loop
    start = time.time()
    while not _deadline_hit():
        _sa_move(centers, angs, T)
        ov = _total_overlap(centers, angs)
        side = _bounding_side(centers, angs, outer_ang)
        # accept
        if ov < best_ov or (ov <= best_ov and side < best_side):
            best_ov = ov
            best_side = side
            best_c = [list(c) for c in centers]
            best_a = list(angs)
        else:
            d = (ov - best_ov) + max(0.0, (side - best_side)) * 0.5
            if random.random() < math.exp(-d / max(T, 1e-9)):
                pass
            else:
                # revert
                centers[i] = [best_c[i][0], best_c[i][1]] if False else centers[i]
        T *= 0.999
        it += 1
        if it % 50 == 0:
            # periodic physics settle to escape local minima
            c2, a2, ov2, side2 = _settle([list(c) for c in centers], list(angs), S, outer_ang, max_iter=60)
            if ov2 < best_ov or (ov2 <= best_ov and side2 < best_side):
                best_ov = ov2
                best_side = side2
                best_c = c2
                best_a = a2
                centers = [list(c) for c in c2]
                angs = list(a2)
    return best_c, best_a, best_side, best_ov


def construct():
    global _START
    _START = time.time()
    random.seed(12345)
    outer_ang = 0.0

    # binary search on S
    lo, hi = 2.0, 12.0
    best_result = None
    # first, find a feasible hi
    for trial in range(3):
        if _deadline_hit():
            break
        c, a, side, ov = _search(hi, outer_ang, 0)
        if ov < 1e-4 and side <= hi + 1e-4:
            best_result = (c, a, side)
            break
        hi += 2.0

    # binary search for minimal S
    for _ in range(14):
        if _deadline_hit():
            break
        mid = 0.5 * (lo + hi)
        c, a, side, ov = _search(mid, outer_ang, 0)
        if ov < 1e-4 and side <= mid + 1e-4:
            hi = mid
            best_result = (c, a, side)
        else:
            lo = mid

    if best_result is None:
        # fallback: return a valid-ish large hexagon
        c, a, side, ov = _search(hi, outer_ang, 0)
        best_result = (c, a, side)

    c, a, side = best_result
    inner = [[c[i][0], c[i][1], math.degrees(a[i])] for i in range(N)]
    return inner, (0.0, 0.0), side, math.degrees(outer_ang)
