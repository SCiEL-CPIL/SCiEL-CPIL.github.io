"""Rewritten search: structured seeds + fast local search to maximize exact-4 lines."""
import time
import random
from math import gcd
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)


def _line_key(a, b):
    """Canonical key for the line through integer points a=(x1,y1), b=(x2,y2).
    Returns (A,B,C) with A*x+B*y=C, gcd(A,B)=1, and a sign convention."""
    x1, y1 = a
    x2, y2 = b
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
    # sign convention: make first non-zero of (A,B) positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _score_and_lines(points):
    """Return (num_exact4_lines, num_lines_with_5plus, line_map).
    line_map maps line_key -> list of indices on that line (len>=3)."""
    n = len(points)
    groups = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            groups[k].append((i, j))
    # Build line -> set of points
    line_pts = defaultdict(set)
    for k, pairs in groups.items():
        for (i, j) in pairs:
            line_pts[k].add(i)
            line_pts[k].add(j)
    exact4 = 0
    fiveplus = 0
    for k, s in line_pts.items():
        cnt = len(s)
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            fiveplus += 1
    return exact4, fiveplus, line_pts


def _valid(points):
    """No five collinear, all distinct, |x|,|y|<=1000."""
    if len(set(map(tuple, points))) != len(points):
        return False
    for (x, y) in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, fiveplus, _ = _score_and_lines(points)
    return fiveplus == 0


def _score(points):
    exact4, fiveplus, _ = _score_and_lines(points)
    if fiveplus > 0:
        return -10**9 - fiveplus
    return exact4


def _neighbors(idx, points, rng, radius):
    """Generate candidate replacements for point idx."""
    x0, y0 = points[idx]
    cands = []
    # random small moves
    for _ in range(40):
        dx = rng.randint(-radius, radius)
        dy = rng.randint(-radius, radius)
        nx, ny = x0 + dx, y0 + dy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            cands.append([nx, ny])
    # a few larger jumps
    for _ in range(10):
        dx = rng.randint(-radius * 6, radius * 6)
        dy = rng.randint(-radius * 6, radius * 6)
        nx, ny = x0 + dx, y0 + dy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            cands.append([nx, ny])
    return cands


def _hill_climb(points, rng, deadline_fn):
    points = [list(p) for p in points]
    cur = _score(points)
    improved = True
    while improved and deadline_fn() > 5.0:
        improved = False
        n = len(points)
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            if deadline_fn() < 5.0:
                return points, cur
            best_new = None
            best_val = cur
            cands = _neighbors(idx, points, rng, 4)
            # dedup
            seen = set()
            for c in cands:
                t = (c[0], c[1])
                if t in seen:
                    continue
                seen.add(t)
                # distinctness check
                if any((p[0] == c[0] and p[1] == c[1]) for j, p in enumerate(points) if j != idx):
                    continue
                old = points[idx]
                points[idx] = c
                v = _score(points)
                if v > best_val:
                    best_val = v
                    best_new = list(c)
                points[idx] = old
            if best_new is not None:
                points[idx] = best_new
                cur = best_val
                improved = True
    return points, cur


def _seed_grid(step, ox, oy, s=4):
    return [[ox + i * step, oy + j * step] for i in range(s) for j in range(s)]


def _seed_affine_grid(rng):
    # 4x4 grid with a random unimodular-ish affine transform + translation
    base = [[i, j] for i in range(4) for j in range(4)]
    # random integer 2x2 with det != 0
    while True:
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        det = a * d - b * c
        if det != 0:
            break
    ox = rng.randint(-50, 50)
    oy = rng.randint(-50, 50)
    pts = []
    for (x, y) in base:
        nx = a * x + b * y + ox
        ny = c * x + d * y + oy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            pts.append([nx, ny])
    return pts


def _seed_perturbed_grid(rng):
    base = _seed_grid(1, 0, 0)
    pts = []
    for (x, y) in base:
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        pts.append([x + dx, y + dy])
    return pts


def _seed_lattice(rng, step):
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i * step + rng.randint(-1, 1), j * step + rng.randint(-1, 1)])
    return pts


def construct():
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = _START + 118.0  # safety margin
    rng = random.Random(12345)

    seeds = []
    # canonical 4x4 grid
    seeds.append(_seed_grid(1, 0, 0))
    # various steps
    for step in (2, 3, 5, 7):
        seeds.append(_seed_grid(step, 0, 0))
    # translations
    seeds.append(_seed_grid(1, 3, 5))
    # affine grids
    for _ in range(30):
        s = _seed_affine_grid(rng)
        if len(s) == 16:
            seeds.append(s)
    # perturbed grids
    for _ in range(30):
        seeds.append(_seed_perturbed_grid(rng))
    # lattice with small step
    for step in (2, 3):
        for _ in range(10):
            seeds.append(_seed_lattice(rng, step))

    best_points = None
    best_score = -10**12

    for s in seeds:
        if _remaining() < 3.0:
            break
        # quick validity
        if len(set(map(tuple, s))) != 16:
            continue
        if any(abs(x) > 1000 or abs(y) > 1000 for (x, y) in s):
            continue
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_points = [list(p) for p in s]
        # hill climb from this seed (only if it's a valid-ish start, i.e. no 5-collinear)
        _, fiveplus, _ = _score_and_lines(s)
        if fiveplus == 0:
            if _remaining() < 2.0:
                break
            pts, sc2 = _hill_climb(s, rng, _remaining)
            if sc2 > best_score:
                best_score = sc2
                best_points = pts
        else:
            # still try to fix by climbing (score heavily penalizes 5+)
            if _remaining() < 2.0:
                break
            pts, sc2 = _hill_climb(s, rng, _remaining)
            if sc2 > best_score:
                best_score = sc2
                best_points = pts

    # Final safety: ensure validity; if not, fall back to canonical grid.
    if best_points is None or not _valid(best_points):
        best_points = [[x, y] for x in range(4) for y in range(4)]

    # Return as list of [x, y]
    return [[int(x), int(y)] for (x, y) in best_points]

# EVOLVE-BLOCK-END
