"""Fresh design: triangular-lattice hexagon packing + tight outer-hexagon fit."""
import math
import time

N = 11
T0 = time.time()
DEADLINE = T0 + 120.0

# Triangular-lattice positions for N unit hexagons (side length 1).
# Neighbour distance for non-overlap = 2*apothem = sqrt(3).
_AX = math.sqrt(3.0)
_AY = 1.5
_pos = []
for i in range(N):
    for j in range(N):
        _pos.append((i * _AX, j * _AY))


def _hex_vertices(cx, cy, s, ang):
    return [(cx + s * math.cos(ang + math.pi / 3.0 * k),
             cy + s * math.sin(ang + math.pi / 3.0 * k)) for k in range(6)]


def _max_vertex_dist(pts, ox, oy):
    m = 0.0
    for (px, py) in pts:
        d2 = (px - ox) ** 2 + (py - oy) ** 2
        if d2 > m:
            m = d2
    return math.sqrt(m)


def _outer_side(pts, ox, oy, ang):
    # Smallest regular hexagon (center o, rotation ang) containing all pts.
    # Radius needed = max dot(p-o, outward_unit_dir) over 6 directions.
    r = 0.0
    for k in range(6):
        a = ang + math.pi / 3.0 * k
        ux, uy = math.cos(a), math.sin(a)
        for (px, py) in pts:
            v = (px - ox) * ux + (py - oy) * uy
            if v > r:
                r = v
    # circumradius = 2 * apothem = 2 * (side * cos(pi/6)) = side * sqrt(3)
    return r / math.sqrt(3.0)


def _side_for(pts, ang):
    # Translate cluster so its bounding box is centred at origin.
    minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
    miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
    ox = 0.5 * (minx + maxx)
    oy = 0.5 * (miny + maxy)
    return _outer_side(pts, ox, oy, ang)


def _nm(f, x0, y0, max_iter=1000, tol=1e-8):
    # 2-variable Nelder-Mead (no reflection of the worst point needed).
    a = 1.5; r = 2.0; c = 0.5
    x = [x0, y0]
    f0 = f(x[0], x[1])
    dx = 0.05
    x1 = [x[0] + dx, x[1]]
    x2 = [x[0], x[1] + dx]
    f1 = f(x1[0], x1[1]); f2 = f(x2[0], x2[1])
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        fs = sorted([(f0, x), (f1, x1), (f2, x2)])
        fa, fb, fc = fs[0][0], fs[1][0], fs[2][0]
        A, B, C = fs[0][1], fs[1][1], fs[2][1]
        if abs(fc - fa) < tol * max(1.0, abs(fa)):
            break
        # centroid of two best
        mx = (A[0] + B[0]) / 2.0
        my = (A[1] + B[1]) / 2.0
        # reflection
        rx = mx + a * (mx - C[0]); ry = my + a * (my - C[1])
        fr = f(rx, ry)
        if fr < fb:
            if fr < fa:
                # expansion
                ex = mx + r * (rx - mx); ey = my + r * (ry - my)
                fe = f(ex, ey)
                if fe < fr:
                    C, fc = [ex, ey], fe
                else:
                    C, fc = [rx, ry], fr
            else:
                C, fc = [rx, ry], fr
        else:
            # contraction
            cx = mx + c * (C[0] - mx); cy = my + c * (C[1] - my)
            fc2 = f(cx, cy)
            if fc2 < fc:
                C, fc = [cx, cy], fc2
            else:
                # shrink toward best
                for P in (B, C):
                    P[0] = A[0] + 0.5 * (P[0] - A[0])
                    P[1] = A[1] + 0.5 * (P[1] - A[1])
                fb = f(B[0], B[1]); fc = f(C[0], C[1])
        x0, x1, x2 = A, B, C
        f0, f1, f2 = fa, fb, fc
    fs = sorted([(f(x[0], x[1]), x)])
    return fs[0][0], fs[0][1]


def construct():
    # Scale cluster down slightly to guarantee a gap (lattice hexagons only
    # touch at edges, so a tiny scale gives ample clearance).
    sc = 1.002
    pts = [(x * sc, y * sc) for (x, y) in _pos]

    # Coarse scan of outer-hexagon rotation.
    best = (1e9, 0.0)
    for k in range(36):
        ang = math.pi / 180.0 * k
        s = _side_for(pts, ang)
        if s < best[0]:
            best = (s, ang)
    s0, a0 = best

    # High-precision Nelder-Mead around the best coarse angle.
    def f(ang):
        return _side_for(pts, ang)
    s1, (ang1, _) = _nm(f, a0 - 0.02, a0 + 0.02)
    if s1 < s0:
        s0, a0 = s1, ang1

    # Final tight polish of the rotation.
    def f2(ang):
        return _side_for(pts, ang)
    s2, (ang2, _) = _nm(f2, a0 - 0.005, a0 + 0.005, max_iter=2000, tol=1e-10)
    if s2 < s0:
        s0, a0 = s2, ang2

    # Center the cluster at origin (bbox centre) so the outer hexagon is centred.
    minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
    miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)

    inner = [[x - cx, y - cy, 0.0] for (x, y) in pts]
    return inner, (0.0, 0.0), s0, math.degrees(a0)
