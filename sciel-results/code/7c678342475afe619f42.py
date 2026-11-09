"""Search for 12 integer points minimizing the number of ordinary (exactly-2) lines."""
import time
import random
import itertools
from fractions import Fraction

_DEADLINE = 110.0  # safety margin under 150s
_start = None


def _count_ordinary(pts):
    """Exact count of lines containing exactly two of the given points."""
    n = len(pts)
    # Map each unordered pair to its line; count how many points lie on each line.
    # Use canonical line representation via (A,B,C) from two points with gcd reduction.
    line_pts = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            # canonical direction
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            # line: dy*x - dx*y = c, c = dy*xi - dx*yi
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            line_pts.setdefault(key, set()).update((i, j))
    ordinary = 0
    for key, s in line_pts.items():
        # s is set of indices on this line (union of pair endpoints)
        if len(s) == 2:
            ordinary += 1
    return ordinary


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _valid(pts):
    if len(pts) != 12:
        return False
    seen = set()
    for x, y in pts:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # not all collinear
    p0 = pts[0]
    for i in range(1, 12):
        p1 = pts[1]
        p2 = pts[i]
        if (p1[0]-p0[0])*(p2[1]-p0[1]) != (p1[1]-p0[1])*(p2[0]-p0[0]):
            return True
    return False


def _score(pts):
    return _count_ordinary(pts)


def _try_config(pts):
    pts = [[int(x), int(y)] for x, y in pts]
    if _valid(pts):
        return pts, _score(pts)
    return None, None


def _search():
    global _start
    _start = time.time()
    best = None
    best_val = None

    def consider(pts):
        nonlocal best, best_val
        if time.time() - _start > _DEADLINE:
            return
        pts2, val = _try_config(pts)
        if pts2 is not None and (best_val is None or val < best_val):
            best = pts2
            best_val = val

    # 1) Hardcoded high-symmetry candidates
    # 3x3 Grid
    consider([[x, y] for x in range(3) for y in range(3)] + [[3, 0], [0, 3], [3, 3]])
    # 4x3 Grid
    consider([[x, y] for x in range(4) for y in range(3)])
    # 3x4 Grid
    consider([[x, y] for x in range(3) for y in range(4)])
    # Hesse-like candidate
    hesse_candidate = [
        [0, 0], [1, 0], [0, 1], [1, 1],
        [2, 0], [0, 2], [2, 1], [1, 2], [2, 2],
        [3, 1], [1, 3], [3, 3]
    ]
    consider(herse_candidate)
    
    # Another candidate: 2x6 grid?
    consider([[x, y] for x in range(6) for y in range(2)])
    
    # 2) Local search from best candidate
    if best is not None:
        cur = best
        cur_val = best_val
    else:
        cur = [[x, y] for x in range(3) for y in range(3)] + [[3, 0], [0, 3], [3, 3]]
        cur_val = _score(cur)
        consider(cur)

    box = 10
    rng = random.Random(12345)
    
    def rand_pt():
        return [rng.randint(-box, box), rng.randint(-box, box)]

    improve = True
    stall = 0
    while improve and time.time() - _start < _DEADLINE:
        improve = False
        # try random single-point moves
        for _ in range(1000):
            if time.time() - _start > _DEADLINE:
                break
            idx = rng.randrange(12)
            newp = cur[:]
            newp[idx] = rand_pt()
            c, v = _try_config(newp)
            if c is not None and v < cur_val:
                cur, cur_val = c, v
                consider(c)
                improve = True
                stall = 0
                break
        # try small perturbations of a single point
        for _ in range(1000):
            if time.time() - _start > _DEADLINE:
                break
            idx = rng.randrange(12)
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            newp = cur[:]
            newp[idx] = [cur[idx][0] + dx, cur[idx][1] + dy]
            c, v = _try_config(newp)
            if c is not None and v < cur_val:
                cur, cur_val = c, v
                consider(c)
                improve = True
                stall = 0
                break
        if not improve:
            stall += 1
            if stall > 5:
                break

    if best is None:
        best = [[x, y] for x in range(4) for y in range(3)]
    return best


def construct():
    global _start
    _start = time.time()
    try:
        res = _search()
        if res is None:
            res = [[x, y] for x in range(4) for y in range(3)]
        # final safety: ensure valid
        if not _valid(res):
            res = [[x, y] for x in range(4) for y in range(3)]
        return res
    except Exception:
        return [[x, y] for x in range(4) for y in range(3)]
