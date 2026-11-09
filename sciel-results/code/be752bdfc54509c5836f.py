import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150s kill

def _count_ordinary(points):
    n = len(points)
    seen = set()
    ordinary = 0
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            key = (xi, yi, dx, dy)
            if key in seen:
                continue
            seen.add(key)
            cnt = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if dx * (yk - yi) == dy * (xk - xi):
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary

def _not_all_collinear(points):
    for i in range(1, len(points)):
        for j in range(i + 1, len(points)):
            if (points[i][0] - points[0][0]) * (points[j][1] - points[0][1]) != \
               (points[j][0] - points[0][0]) * (points[i][1] - points[0][1]):
                return True
    return False

def _hex_init():
    pts = [(0, 0)]
    for r in range(1, 3):
        for a in range(6):
            for b in range(6 * r):
                pass
    # Build hex ring points: ring r has 6r points
    for r in range(1, 3):
        # six sides
        for side in range(6):
            # direction vectors for hex lattice
            dirs = [(r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)]
            d = dirs[side]
            for t in range(r):
                # point along this edge
                if side == 0:
                    p = (t + 1, 0)
                elif side == 1:
                    p = (r - t, t + 1)
                elif side == 2:
                    p = (-t - 1, r)
                elif side == 3:
                    p = (-r, r - t - 1)
                elif side == 4:
                    p = (-r + t + 1, -t - 1)
                else:
                    p = (t + 1, -r)
                pts.append(p)
    return [[x, y] for x, y in pts[:12]]

def _directed_move(points):
    """Try to move a point onto an extension of a line through two other points
    to create a 3+ point line, reducing ordinary lines."""
    n = len(points)
    best = None
    best_score = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, yi = points[i]
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # candidate positions along this line: i + k*(dx,dy) for small k
            for k in range(-4, 5):
                if k == 0 or k == 1:
                    continue
                nx = xi + k * dx
                ny = yi + k * dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                # check distinctness
                cand = points[:]
                cand[i] = [nx, ny]
                if len(set(map(tuple, cand))) != n:
                    continue
                if not _not_all_collinear(cand):
                    continue
                s = _count_ordinary(cand)
                if best_score is None or s < best_score:
                    best_score = s
                    best = cand
    return best, best_score

def _random_move(points, rng):
    n = len(points)
    i = rng.randrange(n)
    # random small perturbation
    dx = rng.randint(-3, 3)
    dy = rng.randint(-3, 3)
    if dx == 0 and dy == 0:
        return None
    cand = points[:]
    cand[i] = [cand[i][0] + dx, cand[i][1] + dy]
    if abs(cand[i][0]) > 10**9 or abs(cand[i][1]) > 10**9:
        return None
    if len(set(map(tuple, cand))) != n:
        return None
    if not _not_all_collinear(cand):
        return None
    return cand

def construct():
    global _DEADLINE
    rng = random.Random(12345)
    # start from hex lattice
    best = _hex_init()
    best_score = _count_ordinary(best)
    
    # directed local search
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for _ in range(200):
            if time.time() > _DEADLINE:
                break
            cand, s = _directed_move(best)
            if cand is not None and s < best_score:
                best = cand
                best_score = s
                improved = True
    
    # random perturbation + hill climb to escape local minima
    for _ in range(50):
        if time.time() > _DEADLINE:
            break
        cand = _random_move(best, rng)
        if cand is None:
            continue
        s = _count_ordinary(cand)
        if s < best_score:
            best = cand
            best_score = s
            # local search from new point
            improved = True
            while improved and time.time() < _DEADLINE:
                improved = False
                for _ in range(100):
                    if time.time() > _DEADLINE:
                        break
                    c2, s2 = _directed_move(best)
                    if c2 is not None and s2 < best_score:
                        best = c2
                        best_score = s2
                        improved = True
        elif s == best_score:
            # accept equal to explore
            if rng.random() < 0.3:
                best = cand
                # local search
                improved = True
                while improved and time.time() < _DEADLINE:
                    improved = False
                    for _ in range(100):
                        if time.time() > _DEADLINE:
                            break
                        c2, s2 = _directed_move(best)
                        if c2 is not None and s2 < best_score:
                            best = c2
                            best_score = s2
                            improved = True
    
    return best
