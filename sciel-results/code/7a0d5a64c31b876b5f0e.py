"""Explicit construction for the 12-point, few-ordinary-lines task.

Strategy:
  - Simulated annealing with structured seeds and line-extension moves.
  - Multiple restarts from diverse seeds.
  - Deadline guard to return best configuration within budget.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 120 s kill


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _count_ordinary_fast(points):
    """Faster count using integer arithmetic without gcd normalization where possible."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _make_seed_grid(rows=3, cols=4, ox=0, oy=0, scale=1):
    """rows x cols grid with optional offset and scale."""
    return [[(x * scale + ox), (y * scale + oy)] for x in range(cols) for y in range(rows)]


def _make_seed_lines():
    """Seed: 12 points arranged on structured lines sharing intersections."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [1, 3], [2, 3], [3, 3], [4, 3],
        [0, 3], [1, 2], [2, 1],
        [0, 1]
    ]
    seen = set()
    clean = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            clean.append(p[:])
    return clean


def _make_seed_boroczky():
    """Böröczky-type configuration: points on two lines plus conic.
    
    Use 6 points on y=0, 6 points on y=1, with careful x-coordinates
    to maximize 3-point lines.
    """
    # 6 points on bottom line y=0
    # 6 points on top line y=1
    # Choose x-coords to create diagonal 3-point lines
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
    ]
    return pts


def _make_seed_perturbed_grid():
    """4x3 grid with one point moved to create extra 3-point lines."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
    ]
    # Try moving (3,2) to (4,3) or other positions
    return pts


def _candidate_pool(points, radius=10):
    """Generate candidate integer points near the existing point cloud."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)
    pool = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            pool.append((cx + dx, cy + dy))
    # Points aligned with existing lines (extend each line)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-2, 3):
                pool.append((x1 + k * dx, y1 + k * dy))
                pool.append((x2 + k * dx, y2 + k * dy))
    # Deduplicate
    seen = set()
    unique = []
    for p in pool:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _simulated_annealing(points, max_iters=500000):
    """Simulated annealing to escape local minima."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    rng = random.Random(7)
    T = 2.0
    T_min = 0.01
    cooling = 0.99995
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        if T < T_min:
            T = T_min
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        
        # Generate candidate
        if rng.random() < 0.5:
            # Small perturbation
            nx = old[0] + rng.randint(-4, 4)
            ny = old[1] + rng.randint(-4, 4)
            cand = (nx, ny)
        else:
            # Line extension
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i == j:
                i = (j + 1) % 12
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            k = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
            base = cur[i] if rng.random() < 0.5 else cur[j]
            cand = (base[0] + k * dx, base[1] + k * dy)
        
        if cand in cur_set:
            continue
        if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
            continue
        
        new_pts = [p[:] for p in cur]
        new_pts[idx] = list(cand)
        if not _valid(new_pts):
            continue
        
        new_score = _count_ordinary(new_pts)
        delta = new_score - cur_score
        
        if delta < 0 or (rng.random() < _exp_neg(delta / T)):
            cur = new_pts
            cur_score = new_score
            cur_set = set(tuple(p) for p in cur)
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]
        
        T *= cooling
    
    return best, best_score


def _exp_neg(x):
    """exp(-x) with clamping."""
    if x < -50:
        return 0.0
    import math
    return math.exp(-x)


def _local_search2(points, max_iters=300000):
    """Aggressive local search with dynamic candidate generation."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(123)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        improved = False
        # Strategy 1: small perturbation
        for _ in range(15):
            nx = old[0] + rng.randint(-5, 5)
            ny = old[1] + rng.randint(-5, 5)
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        if improved:
            continue
        # Strategy 2: place on a line through two existing points (extend)
        for _ in range(10):
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i == j:
                continue
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            k = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
            nx = x1 + k * dx
            ny = y1 + k * dy
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
    return cur, cur_score


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    def _consider(pts):
        nonlocal best_pts, best_score
        if not _valid(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Multiple seeds
    seeds = []
    seeds.append(_make_seed_grid(3, 4))
    seeds.append(_make_seed_grid(4, 3))
    seeds.append(_make_seed_lines())
    seeds.append(_make_seed_boroczky())
    seeds.append(_make_seed_perturbed_grid())
    # Scaled grids
    for scale in [2, 3]:
        seeds.append(_make_seed_grid(3, 4, 0, 0, scale))
        seeds.append(_make_seed_grid(4, 3, 0, 0, scale))

    # Evaluate all seeds
    for seed in seeds:
        if len(seed) == 12:
            _consider(seed)

    # Simulated annealing from best seed
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_sa, score_sa = _simulated_annealing(best_pts, max_iters=400000)
        if score_sa < best_score:
            best_score = score_sa
            best_pts = pts_sa

    # Local search 2 from best so far
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_ls, score_ls = _local_search2(best_pts, max_iters=200000)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # Another SA round from best
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_sa2, score_sa2 = _simulated_annealing(best_pts, max_iters=300000)
        if score_sa2 < best_score:
            best_score = score_sa2
            best_pts = pts_sa2

    # Final local search
    if best_pts is not None and time.time() < _DEADLINE - 8:
        pts_final, score_final = _local_search2(best_pts, max_iters=150000)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = _make_seed_grid(3, 4)

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
