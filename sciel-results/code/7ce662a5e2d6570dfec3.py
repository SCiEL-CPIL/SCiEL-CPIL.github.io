"""Packing 12 unit hexagons in a regular hexagon: compact 3x4 lattice block + optimized enclosure."""
import math
import time

N = 12
_T0 = time.time()
_DEADLINE = _T0 + 140.0   # safety margin before the 150s wall-clock kill

# --- geometry of a regular hexagon (circumradius = side s) ---
# vertices at angles a + k*60deg, k=0..5
_VA = [math.radians(60.0 * k) for k in range(6)]
_COS = [math.cos(a) for a in _VA]
_SIN = [math.sin(a) for a in _VA]


def _verts(s, a):
    """List of 6 vertex (x,y) for hexagon side s, rotation a (radians)."""
    return [(s * c, s * d) for c, d in zip(_COS, _SIN)]


def _hex_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _poly_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _poly_intersect(A, B):
    """Sutherland-Hodgman polygon-polygon intersection (A clipped by B)."""
    out = list(A)
    nB = len(B)
    for i in range(nB):
        if not out:
            return []
        x1, y1 = B[i]
        x2, y2 = B[(i + 1) % nB]
        ex, ey = x2 - x1, y2 - y1
        nx, ny = ey, -ex          # normal
        c = -(nx * x1 + ny * y1)
        res = []
        nA = len(out)
        for j in range(nA):
            ax, ay = out[j]
            bx, by = out[(j + 1) % nA]
            fa = nx * ax + ny * ay + c
            fb = nx * bx + ny * by + c
            ia = fa >= -1e-12
            ib = fb >= -1e-12
            if ia:
                res.append((ax, ay))
            if ia != ib:
                if abs(fa - fb) > 1e-15:
                    t = fa / (fa - fb)
                    res.append((ax + t * (bx - ax), ay + t * (by - ay)))
        out = res
    return out


def _hex_in_hex_area(A, B):
    """Area of intersection of convex hexagons A and B (each 6 pts, CCW)."""
    # quick reject by bounding boxes
    ax = [p[0] for p in A]; ay = [p[1] for p in A]
    bx = [p[0] for p in B]; by = [p[1] for p in B]
    if min(ax) > max(bx) or min(bx) > max(ax):
        return 0.0
    if min(ay) > max(by) or min(by) > max(ay):
        return 0.0
    inter = _poly_intersect(A, B)
    if len(inter) < 3:
        return 0.0
    return _poly_area(inter)


def _hex_overlap_penalty(A, B, eps):
    """Smooth surrogate for overlap: 0 if disjoint, else (area - eps)^2."""
    area = _hex_in_hex_area(A, B)
    if area <= eps:
        return 0.0
    d = area - eps
    return d * d


def _hexes_overlap(verts_list, eps):
    """True if any pair of hexagons overlaps beyond tolerance eps."""
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_in_hex_area(verts_list[i], verts_list[j]) > eps:
                return True
    return False


def _tight_L(pts, a, ox, oy):
    """Minimal enclosing regular-hexagon side length (rotation a) for points
    (shifted by (ox,oy)). Hexagon centered at origin. Uses 6 support functions."""
    best = 0.0
    for k in range(6):
        c = _COS[k]; s = _SIN[k]
        sup = 0.0
        for (x, y) in pts:
            v = (x + ox) * c + (y + oy) * s
            if v > sup:
                sup = v
        if sup > best:
            best = sup
    return best


def _enclosure_cost(pts, L, a, ox, oy, eps):
    """Penalty if the enclosing hexagon (side L, rotation a, center (-ox,-oy))
    fails to contain all points (each point must have support <= L)."""
    # hexagon centered at origin has support L in each of the 6 directions.
    # A point (x+ox, y+oy) is inside iff for all k: (x+ox)*c + (y+oy)*s <= L
    # (and the opposite is covered by k+3). We only need the 6 outward normals.
    pen = 0.0
    for k in range(6):
        c = _COS[k]; s = _SIN[k]
        sup = 0.0
        for (x, y) in pts:
            v = (x + ox) * c + (y + oy) * s
            if v > sup:
                sup = v
        over = sup - L
        if over > eps:
            pen += over * over
    return pen


# --- fixed inner hexagon layout: 3 rows x 4 columns on hex lattice ---
# hex centers: column i in {-1.5,-0.5,0.5,1.5}, row j in {-sqrt3,0,sqrt3}
_SQRT3 = math.sqrt(3.0)
_BASE_PTS = []
for j in (-_SQRT3, 0.0, _SQRT3):
    for i in (-1.5, -0.5, 0.5, 1.5):
        _BASE_PTS.append((i, j))
assert len(_BASE_PTS) == N


def _base_verts(s=1.0, a=0.0):
    return [_verts(s, a) for _ in _BASE_PTS]


def construct():
    """Return (inner, center, side, angle). inner rows (x,y,angle_deg); outer
    center (x,y), side length, rotation (degrees). Score = outer side length."""
    # Start from the compact 3x4 block; enclosing hexagon centered near origin.
    base_v = _base_verts(1.0, 0.0)
    L0 = _tight_L(_BASE_PTS, 0.0, 0.0, 0.0)
    L0 *= 1.02
    best = (L0, 0.0, 0.0, 0.0)  # (L, a, ox, oy)

    # Nelder-Mead on (L, a, ox, oy) minimizing L + overlap_penalty + enclosure_penalty
    def obj(v):
        L, a, ox, oy = v
        if L < 0.5:
            return 1e6
        # inner hexagons (unit, no rotation) shifted by (ox,oy)
        vs = []
        for (px, py) in _BASE_PTS:
            cx = px + ox
            cy = py + oy
            vs.append([(cx + x, cy + y) for (x, y) in _verts(1.0, 0.0)])
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                pen += _hex_overlap_penalty(vs[i], vs[j], 1e-9)
        # enclosure: hexagon side L, rotation a, center (-ox,-oy)
        pen += _enclosure_cost(_BASE_PTS, L, a, ox, oy, 1e-9)
        return L + 10.0 * pen

    # initial simplex
    step = 0.05
    x0 = [L0, 0.0, 0.0, 0.0]
    simplex = [list(x0)]
    simplex.append([L0 + step, 0.0, 0.0, 0.0])
    simplex.append([L0, step, 0.0, 0.0])
    simplex.append([L0, 0.0, step, 0.0])
    simplex.append([L0, 0.0, 0.0, step])
    f = [obj(s) for s in simplex]

    def _swap_simplex(simplex, f, i, j):
        simplex[i], simplex[j] = simplex[j], simplex[i]
        f[i], f[j] = f[j], f[i]

    iters = 0
    max_iters = 4000
    while iters < max_iters:
        if time.time() > _DEADLINE:
            break
        iters += 1
        order = sorted(range(5), key=lambda i: f[i])
        _swap_simplex(simplex, f, order[0], 0)
        _swap_simplex(simplex, f, order[4], 4)
        # reflect
        c = 1.0
        xr = [x0[0] for x0 in simplex[0]]
        for i in range(4):
            xr[i] = simplex[0][i] + c * (simplex[0][i] - simplex[4][i])
        fr = obj(xr)
        if f[0] <= fr < f[3]:
            simplex[4] = xr; f[4] = fr
            continue
        # expand
        if fr < f[0]:
            xe = [simplex[0][i] + 2.0 * (xr[i] - simplex[0][i]) for i in range(4)]
            fe = obj(xe)
            if fe < fr:
                simplex[4] = xe; f[4] = fe
            else:
                simplex[4] = xr; f[4] = fr
            continue
        # contract
        if fr >= f[4]:
            cc = 0.5
            xc = [simplex[0][i] + cc * (simplex[4][i] - simplex[0][i]) for i in range(4)]
            fc = obj(xc)
            if fc < f[4]:
                simplex[4] = xc; f[4] = fc
                continue
            # shrink
            for i in range(1, 5):
                for k in range(4):
                    simplex[i][k] = simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k])
                f[i] = obj(simplex[i])
            continue
        else:
            cc = 0.5
            xc = [simplex[0][i] + cc * (xr[i] - simplex[0][i]) for i in range(4)]
            fc = obj(xc)
            if fc < fr:
                simplex[4] = xc; f[4] = fc
            else:
                for i in range(1, 5):
                    for k in range(4):
                        simplex[i][k] = simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k])
                    f[i] = obj(simplex[i])
            continue
        # track best
        bi = min(range(5), key=lambda i: f[i])
        if f[bi] < best[0] + 1e-12:
            pass
    # after loop, best of simplex
    bi = min(range(5), key=lambda i: f[i])
    Lb, ab, oxb, oyb = simplex[bi]
    if Lb < best[0]:
        best = (Lb, ab, oxb, oyb)

    # polish: refine L to the exact tight value for the chosen (a, ox, oy)
    L, a, ox, oy = best
    L = _tight_L(_BASE_PTS, a, ox, oy)
    # verify feasibility (no overlap, all inside); if not, nudge L up slightly
    vs = []
    for (px, py) in _BASE_PTS:
        cx = px + ox
        cy = py + oy
        vs.append([(cx + x, cy + y) for (x, y) in _verts(1.0, 0.0)])
    if _hexes_overlap(vs, 1e-7):
        # overlap in the compact block shouldn't happen; guard by expanding L
        L *= 1.001
    # enclosure check
    if _enclosure_cost(_BASE_PTS, L, a, ox, oy, 1e-9) > 0:
        L = _tight_L(_BASE_PTS, a, ox, oy) * 1.0001

    # outer hexagon center = (-ox, -oy)
    center = (-ox, -oy)
    angle_deg = math.degrees(a)

    inner = [[(px + ox), (py + oy), 0.0] for (px, py) in _BASE_PTS]
    return inner, center, L, angle_deg
