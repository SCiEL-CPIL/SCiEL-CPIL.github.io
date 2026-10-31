import time

# EVOLVE-BLOCK-START
import math
import numpy as np

_AX = np.array([1.0, 0.0])
_AY = np.array([0.5, math.sqrt(3.0) / 2.0])
_D3 = math.sqrt(3.0)


def _hex_verts(cx, cy, side, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        x = side * math.cos(t)
        y = side * math.sin(t)
        out.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return out


def _poly_area(p):
    s = 0.0
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def _seg_cross(ax, ay, bx, by, cx, cy, dx, dy):
    def o(p, q, r):
        v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 1 if v > 1e-12 else (-1 if v < -1e-12 else 0)
    o1, o2 = o((ax, ay), (bx, by), (cx, cy)), o((ax, ay), (bx, by), (dx, dy))
    o3, o4 = o((cx, cy), (dx, dy), (ax, ay)), o((cx, cy), (dx, dy), (bx, by))
    return o1 != o2 and o3 != o4


def _seg_dist(p, q, r, s):
    if _seg_cross(p[0], p[1], q[0], q[1], r[0], r[1], s[0], s[1]):
        return 0.0
    return min(math.hypot(p[0] - r[0], p[1] - r[1]), math.hypot(p[0] - s[0], p[1] - s[1]),
               math.hypot(q[0] - r[0], q[1] - r[1]), math.hypot(q[0] - s[0], q[1] - s[1]))


def _hex_gap(h1, h2):
    m = 1e9
    n = len(h1)
    for i in range(n):
        m = min(m, _seg_dist(h1[i], h1[(i + 1) % n], h2[i], h2[(i + 1) % n]))
    return m


def _min_gap(inner, outer, outer_side):
    og = _hex_verts(outer[0], outer[1], outer_side, 0.0)
    m = 1e9
    for x, y, _a in inner:
        m = min(m, _hex_gap(_hex_verts(x, y, 1.0, 0.0), og))
    return m


def _max_count(eps):
    # feasible center region: hexagon of radius R = side - 1 (in axial units)
    R = eps
    if R <= 0:
        return 0
    best = 0
    for i in range(int(R) + 2):
        for j in range(int(R) + 2):
            for k in range(int(R) + 2):
                if i + j + k == int(R):
                    cnt = 2 * (i * j + j * k + k * i) + i + j + k + 1
                    if cnt > best:
                        best = cnt
    return best


def _greedy(eps, cap=4000):
    R = eps
    if R <= 0:
        return []
    Rf = int(math.floor(R))
    cands = []
    for i in range(Rf + 1):
        for j in range(Rf + 1):
            for k in range(Rf + 1):
                if i + j + k == Rf:
                    cands.append((i, j, k))
    if len(cands) > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cands), size=cap, replace=False)
        cands = [cands[t] for t in idx]
    cands.sort(key=lambda t: -(t[0] * t[1] + t[1] * t[2] + t[2] * t[0]))
    placed = []
    used = set()
    for (i, j, k) in cands:
        ok = True
        for (a, b, c) in placed:
            if abs(i - a) + abs(j - b) + abs(k - c) <= 1:
                ok = False
                break
        if ok:
            x = _AX[0] * i + _AY[0] * j
            y = _AX[1] * i + _AY[1] * j
            placed.append((i, j, k))
            used.add((x, y))
    return [(x, y) for (x, y) in used]


def construct():
    t0 = time.time()
    deadline = 120.0
    best = None

    for n in range(1, 200000):
        if time.time() > deadline - 30.0:
            break
        eps = (n - 1) / 2.0
        mc = _max_count(eps)
        if mc < n:
            break
        if mc > n:
            continue
        if n > 4000:
            continue
        inner = _greedy(eps, cap=4000)
        if len(inner) != n:
            continue
        inner = [[x, y, 0.0] for (x, y) in inner]
        side = eps + 1.0
        gap = _min_gap(inner, (0.0, 0.0), side)
        if gap < 1e-9:
            continue
        if best is None or side < best[2] - 1e-12:
            best = (inner, (0.0, 0.0), side, 0.0)
        if time.time() > deadline - 30.0:
            break

    if best is None:
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 2.0, 0.0
    return best
# EVOLVE-BLOCK-END
