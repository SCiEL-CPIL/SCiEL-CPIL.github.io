"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _det3(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def _check_valid(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_square(_sq_dist(points[i], points[j])):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _det3(points[i][0], points[i][1],
                         points[j][0], points[j][1],
                         points[k][0], points[k][1]) == 0:
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # Circle equation x^2 + y^2 + Ax + By + C = 0
                    # 2x*A + 2y*B + C = -(x^2 + y^2)
                    def row(x, y):
                        return (2 * x, 2 * y, 1, -(x * x + y * y))
                    r1 = row(x1, y1)
                    r2 = row(x2, y2)
                    r3 = row(x3, y3)
                    r4 = row(x4, y4)
                    # 4x4 determinant
                    a, b, c, d = r1
                    e, f, g, h = r2
                    i_, j_, k_, l_ = r3
                    m, n_, o, p = r4
                    det = (a * (f * (k_ * p - l_ * o) - g * (j_ * p - l_ * n_) + h * (j_ * o - k_ * n_))
                           - b * (e * (k_ * p - l_ * o) - g * (i_ * p - l_ * m) + h * (i_ * o - k_ * m))
                           + c * (e * (j_ * p - l_ * n_) - f * (i_ * p - l_ * m) + h * (i_ * n_ - j_ * m))
                           - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
                    if det == 0:
                        return False
    return True

def _try_add(points, cand):
    new = points + [cand]
    return _check_valid(new)

def _circles_around(pts, R):
    cands = set()
    for (a, b) in pts:
        for r in range(1, R + 1):
            r2 = r * r
            for dx in range(-r, r + 1):
                dy2 = r2 - dx * dx
                if dy2 < 0:
                    continue
                dy = math.isqrt(dy2)
                if dy * dy != dy2:
                    continue
                cands.add((a + dx, b + dy))
                if dy != 0:
                    cands.add((a + dx, b - dy))
    return cands

def _greedy_extend(pts, R, deadline, seed=0):
    pts = list(pts)
    rng = random.Random(seed)
    best = list(pts)
    while time.time() < deadline:
        cands = _circles_around(pts, R)
        cands = [c for c in cands if c not in pts]
        rng.shuffle(cands)
        added = False
        for c in cands:
            if time.time() > deadline:
                break
            if _try_add(pts, c):
                pts.append(c)
                added = True
                if len(pts) > len(best):
                    best = list(pts)
        if not added:
            break
    return best

def construct():
    start = time.time()
    deadline = start + 130  # safety margin under 150s kill

    initial = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(initial)

    def maybe_update(p):
        nonlocal best
        if len(p) > len(best):
            best = list(p)

    # 1) Extend the known 5-point set
    for R in (40, 80, 150):
        if time.time() > deadline:
            break
        p = _greedy_extend(initial, R, deadline, seed=100 + R)
        maybe_update(p)

    # 2) Targeted extension: fix a subset of best, regrow the rest
    if len(best) >= 5:
        for keep in range(3, min(len(best), 6) + 1):
            if time.time() > deadline:
                break
            for combo in range(0, 3):
                if time.time() > deadline:
                    break
                rng = random.Random(2000 + keep * 10 + combo)
                idx = rng.sample(range(len(best)), keep)
                base = [best[i] for i in sorted(idx)]
                if not _check_valid(base):
                    continue
                p = _greedy_extend(base, 120, deadline, seed=3000 + keep * 10 + combo)
                maybe_update(p)

    # 3) Fresh builds from a single origin with a few seeds, several radii
    for seed in range(0, 12):
        if time.time() > deadline:
            break
        rng = random.Random(5000 + seed)
        base = [(0, 0)]
        # add a couple of random integer-distance points to seed growth
        for _ in range(3):
            r = rng.randint(10, 60)
            r2 = r * r
            dx = rng.randint(-r, r)
            dy2 = r2 - dx * dx
            dy = math.isqrt(dy2)
            if dy * dy != dy2:
                continue
            cand = (dx, dy) if dy else (dx, 0)
            if cand != (0, 0) and _try_add(base, cand):
                base.append(cand)
        for R in (50, 120):
            if time.time() > deadline:
                break
            p = _greedy_extend(base, R, deadline, seed=6000 + seed * 2 + R)
            maybe_update(p)

    # Ensure the 5-point set is always valid as a fallback
    if not _check_valid(best):
        best = list(initial)

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
