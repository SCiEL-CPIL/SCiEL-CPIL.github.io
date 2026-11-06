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

def _det4_concyclic(p1, p2, p3, p4):
    """Check if 4 points are concyclic using 4x4 determinant."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Matrix rows: [2x, 2y, 1, -(x^2+y^2)]
    # Or simpler: determinant of
    # | x^2+y^2, x, y, 1 |
    # | ... |
    
    # Let's use the standard determinant for concyclicity:
    # | x1^2+y1^2, x1, y1, 1 |
    # | x2^2+y2^2, x2, y2, 1 |
    # | x3^2+y3^2, x3, y3, 1 |
    # | x4^2+y4^2, x4, y4, 1 |
    
    a1 = x1*x1 + y1*y1
    a2 = x2*x2 + y2*y2
    a3 = x3*x3 + y3*y3
    a4 = x4*x4 + y4*y4
    
    # det = 
    # a1*(x2*(y3*1 - 1*y4) - y2*(x3*1 - 1*x4) + 1*(x3*y4 - y3*x4))
    # - x1*(a2*(y3*1 - 1*y4) - y2*(a3*1 - 1*a4) + 1*(a3*y4 - y3*a4))
    # + y1*(a2*(x3*1 - 1*x4) - x2*(a3*1 - 1*a4) + 1*(a3*x4 - x3*a4))
    # - 1*(a2*(x3*y4 - y3*x4) - x2*(a3*y4 - y3*a4) + y2*(a3*x4 - x3*a4))
    
    # This is getting messy to expand manually. Let's just compute the 4x4 det directly.
    # M = [[a1, x1, y1, 1],
    #      [a2, x2, y2, 1],
    #      [a3, x3, y3, 1],
    #      [a4, x4, y4, 1]]
    
    m00, m01, m02, m03 = a1, x1, y1, 1
    m10, m11, m12, m13 = a2, x2, y2, 1
    m20, m21, m22, m23 = a3, x3, y3, 1
    m30, m31, m32, m33 = a4, x4, y4, 1
    
    det = (m00 * (m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31))
         - m01 * (m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30))
         + m02 * (m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30))
         - m03 * (m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30)))
         
    return det == 0

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
                    if _det4_concyclic(points[i], points[j], points[k], points[l]) == 0:
                        return False
    return True

def _can_add(points, cand):
    """Incremental check: can cand be added to points?"""
    n = len(points)
    
    # 1. Check distances
    for i in range(n):
        if not _is_square(_sq_dist(points[i], cand)):
            return False
    
    # 2. Check collinearity with any pair
    for i in range(n):
        for j in range(i + 1, n):
            if _det3(points[i][0], points[i][1],
                     points[j][0], points[j][1],
                     cand[0], cand[1]) == 0:
                return False
    
    # 3. Check concyclicity with any triple
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _det4_concyclic(points[i], points[j], points[k], cand) == 0:
                    return False
    return True

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
    
    # Pre-compute existing points for faster access?
    # The _can_add is the bottleneck.
    
    while time.time() < deadline:
        cands = _circles_around(pts, R)
        cands = [c for c in cands if c not in pts]
        if not cands:
            break
        rng.shuffle(cands)
        added = False
        for c in cands:
            if time.time() > deadline:
                break
            if _can_add(pts, c):
                pts.append(c)
                added = True
                if len(pts) > len(best):
                    best = list(pts)
                if len(pts) >= 12:
                    return best
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
            for combo in range(0, 5): # Increased combos
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
    for seed in range(0, 20): # Increased seeds
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
            if cand != (0, 0) and _can_add(base, cand):
                base.append(cand)
        for R in (50, 120):
            if time.time() > deadline:
                break
            p = _greedy_extend(base, R, deadline, seed=6000 + seed * 2 + R)
            maybe_update(p)

    # 4) Random restarts with random initial sets
    for _ in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(time.time_ns())
        base = []
        for _ in range(3):
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            cand = (x, y)
            if cand not in base and _can_add(base, cand):
                base.append(cand)
        if len(base) >= 3:
            p = _greedy_extend(base, 100, deadline, seed=rng.randint(0, 10000))
            maybe_update(p)

    # Ensure the 5-point set is always valid as a fallback
    if not _check_valid(best):
        best = list(initial)

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
