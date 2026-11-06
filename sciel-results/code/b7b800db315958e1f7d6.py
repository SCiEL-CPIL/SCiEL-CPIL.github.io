import math
import random
import time

N = 20
MAXC = 100
DEADLINE = 140.0  # wall-clock safety margin before the 150s kill


def _norm(dx, dy):
    g = math.gcd(dx, dy)
    if g > 1:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy)


def _line_key(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    dx, dy = _norm(dx, dy)
    # anchor = lexicographically smallest point on the line
    if ax < bx or (ax == bx and ay < by):
        return (ax, ay, dx, dy)
    return (bx, by, dx, dy)


def _lines(points):
    d = {}
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            k = _line_key(xi, yi, xj, yj)
            c = d.get(k)
            if c is None:
                d[k] = (xi, yi, xj, yj)
            else:
                d[k] = (c[0], c[1], c[2], c[3])
    return d


def _max3(d):
    m = 0
    for v in d.values():
        if len(v) >= 3:
            m = 3
            break
    return m


def _count3(d):
    c = 0
    for v in d.values():
        L = len(v)
        if L >= 3:
            c += L * (L - 1) * (L - 2) // 6
    return c


def _has4(d):
    for v in d.values():
        if len(v) >= 4:
            return True
    return False


def _indep(points):
    # exact branch and bound: largest subset with no three collinear
    n = len(points)
    d = _lines(points)
    m3 = _max3(d)
    if m3 == 0:
        return n
    best = [m3]
    stack = [(0, frozenset())]
    while stack:
        i, S = stack.pop()
        if i >= n:
            if len(S) > best[0]:
                best[0] = len(S)
            continue
        if len(S) + (n - i) <= best[0]:
            continue
        xi, yi = points[i]
        bad = False
        for (x1, y1, x2, y2) in S:
            if _line_key(xi, yi, x1, y1) == _line_key(x2, y2, x1, y1):
                bad = True
                break
        if not bad:
            stack.append((i + 1, S | frozenset([(xi, yi)])))
        stack.append((i + 1, S))
    return best[0]


def construct():
    t0 = time.time()
    rng = random.Random(20240517)
    deadline = t0 + DEADLINE

    def remaining():
        return deadline - time.time()

    # random starting configuration with no four collinear
    def random_config():
        pts = []
        used = set()
        while len(pts) < N:
            x = rng.randint(-MAXC, MAXC)
            y = rng.randint(-MAXC, MAXC)
            if (x, y) in used:
                continue
            d = _lines(pts + [(x, y)])
            if _has4(d):
                continue
            pts.append((x, y))
            used.add((x, y))
        return pts

    cfg = random_config()
    d = _lines(cfg)
    c3 = _count3(d)
    best_cfg = list(cfg)
    best_c3 = c3
    best_indep = _indep(best_cfg)
    T = 3.0
    T_end = 0.05
    step = 0.9995

    while T > T_end and remaining() > 3.0:
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i == j:
            continue
        cfg[i], cfg[j] = cfg[j], cfg[i]
        d2 = _lines(cfg)
        if _has4(d2):
            cfg[i], cfg[j] = cfg[j], cfg[i]
            continue
        c3n = _count3(d2)
        if c3n <= c3 or rng.random() < math.exp((c3 - c3n) / T):
            c3 = c3n
            if c3 < best_c3:
                best_c3 = c3
                best_cfg = list(cfg)
        else:
            cfg[i], cfg[j] = cfg[j], cfg[i]
        T *= step

    # final polish: try to lower the exact independence number via targeted swaps
    while remaining() > 2.0:
        improved = False
        for i in range(N):
            if remaining() < 2.0:
                break
            for j in range(i + 1, N):
                if remaining() < 2.0:
                    break
                cfg[i], cfg[j] = cfg[j], cfg[i]
                d2 = _lines(cfg)
                if _has4(d2):
                    cfg[i], cfg[j] = cfg[j], cfg[i]
                    continue
                c3n = _count3(d2)
                if c3n <= best_c3:
                    cur_indep = _indep(cfg)
                    if cur_indep < best_indep:
                        best_indep = cur_indep
                        best_cfg = list(cfg)
                        best_c3 = c3n
                        improved = True
                        c3 = c3n
                        continue
                cfg[i], cfg[j] = cfg[j], cfg[i]
        if not improved and best_c3 <= 0:
            break

    # verify the best configuration is valid and score it exactly
    d_best = _lines(best_cfg)
    if _has4(d_best):
        # should not happen; fall back to parabola
        best_cfg = [(x, x * x) for x in range(-10, 10)]
    if remaining() > 1.0:
        best_indep = _indep(best_cfg)

    return [[x, y] for (x, y) in best_cfg]
