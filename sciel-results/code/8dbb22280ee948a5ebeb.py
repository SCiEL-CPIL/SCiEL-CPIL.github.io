"""Place 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np

N = 12
_DEADLINE = time.time() + 120.0
_MARGIN = 15.0
_SQRT3 = math.sqrt(3.0)


def _hex_vertices(cx, cy, ang_deg, side=1.0):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    v = np.zeros((6, 2))
    for i in range(6):
        t = math.radians(60.0 * i)
        x = side * math.cos(t)
        y = side * math.sin(t)
        v[i, 0] = cx + x * ca - y * sa
        v[i, 1] = cy + x * sa + y * ca
    return v


def _outer_vertices(cxo, cyo, side, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    v = np.zeros((6, 2))
    for i in range(6):
        t = math.radians(60.0 * i)
        x = side * math.cos(t)
        y = side * math.sin(t)
        v[i, 0] = cxo + x * ca - y * sa
        v[i, 1] = cyo + x * sa + y * ca
    return v


def _seg_intersect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return False
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    return (-1e-9 <= t <= 1 + 1e-9) and (-1e-9 <= u <= 1 + 1e-9)


def _polys_intersect(a, b):
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if _seg_intersect(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return True
    return False


def _point_in_poly(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def _overlap(i1, i2, ang1, ang2, eps=1e-4):
    a = _hex_vertices(i1[0], i1[1], ang1)
    b = _hex_vertices(i2[0], i2[1], ang2)
    if _polys_intersect(a, b):
        return True
    if _point_in_poly(a[0], b):
        return True
    if _point_in_poly(b[0], a):
        return True
    return False


def _outer_side_for(inner, angs, cx, cy, rot):
    best = 0.0
    for i in range(N):
        v = _hex_vertices(inner[i][0], inner[i][1], angs[i])
        for p in v:
            dx, dy = p[0] - cx, p[1] - cy
            r = math.hypot(dx, dy)
            th = math.atan2(dy, dx) - math.radians(rot)
            s = _SQRT3 / 2.0 / math.cos((th % (math.pi / 3.0)) - math.pi / 6.0)
            if s > best:
                best = s
    return best


def _objective(x):
    inner = [[x[3 * i], x[3 * i + 1]] for i in range(N)]
    angs = [x[3 * i + 2] for i in range(N)]
    cx, cy, rot = x[3 * N], x[3 * N + 1], x[3 * N + 2]
    for i in range(N):
        for j in range(i + 1, N):
            if _overlap(inner[i], inner[j], angs[i], angs[j]):
                return 1e6
    return _outer_side_for(inner, angs, cx, cy, rot)


def _nelder_mead(f, x0, maxiter=1500, step=0.6):
    n = len(x0)
    x = list(x0)
    fx = f(x)
    simplex = [x[:]]
    vals = [fx]
    for k in range(1, n + 1):
        cand = x0[:]
        cand[k - 1] += step
        simplex.append(cand)
        vals.append(f(cand))
    for _ in range(maxiter):
        if time.time() > _DEADLINE - _MARGIN:
            break
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = [sum(simplex[i][d] for i in range(n)) / n for d in range(n)]
        xr = [centroid[d] + 1.0 * (centroid[d] - simplex[-1][d]) for d in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [centroid[d] + 2.0 * (centroid[d] - simplex[-1][d]) for d in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = [centroid[d] + 0.5 * (simplex[-1][d] - centroid[d]) for d in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][d] + 0.5 * (simplex[i][d] - simplex[0][d]) for d in range(n)]
                    vals[i] = f(simplex[i])
    bi = min(range(n + 1), key=lambda i: vals[i])
    return simplex[bi], vals[bi]


def _base_layout():
    # hex-grid coordinates (i, j) -> flat-top centers
    cells = [(0, 0), (1, 0), (2, 0), (3, 0),
             (0, 1), (1, 1), (2, 1),
             (0, -1), (1, -1), (2, -1),
             (1, 2), (1, -2)]
    pts = []
    for (i, j) in cells:
        x = 1.5 * i
        y = (_SQRT3 / 2.0) * (j + i / 2.0)
        pts.append([x, y, 0.0])
    return pts


def construct():
    inner = _base_layout()
    angs = [0.0] * N
    cx, cy, rot = 0.0, 0.0, 0.0
    x0 = []
    for p in inner:
        x0 += [p[0], p[1], p[2]]
    x0 += [cx, cy, rot]
    try:
        best_x, best_f = _nelder_mead(_objective, x0, maxiter=1200, step=0.5)
        if time.time() < _DEADLINE - _MARGIN:
            x2 = best_x[:]
            for i in range(N):
                for d in range(3):
                    x2[3 * i + d] += (1.0 if (i + d) % 2 == 0 else -1.0) * 0.25
            x2 += [0.0, 0.0, 0.0]
            x2[3 * N] = best_x[3 * N]; x2[3 * N + 1] = best_x[3 * N + 1]; x2[3 * N + 2] = best_x[3 * N + 2]
            rx, rf = _nelder_mead(_objective, x2, maxiter=800, step=0.2)
            if rf < best_f:
                best_x, best_f = rx, rf
    except Exception:
        pass
    if best_f >= 1e5:
        # fallback: keep base layout, compute side
        best_x = x0
        best_f = _objective(best_x)
    inner_out = [[best_x[3 * i], best_x[3 * i + 1], best_x[3 * i + 2]] for i in range(N)]
    return inner_out, (best_x[3 * N], best_x[3 * N + 1]), best_f, best_x[3 * N + 2]
