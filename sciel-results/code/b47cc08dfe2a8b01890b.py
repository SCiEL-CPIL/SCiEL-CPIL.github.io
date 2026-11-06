"""Improved program: local-search maximization of exact-4 lines from a good seed."""
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150 s wall-clock kill

def _four_lines(points):
    """Count lines containing exactly four of the points; return (count, max_collinear)."""
    lines = defaultdict(set)
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx, dy = xj - xi, yj - yi
            if dx == 0 and dy == 0:
                continue
            lines[(dx, dy, xi, yi)].add(i)
            lines[(dx, dy, xj, yj)].add(i)
    count = 0
    maxc = 0
    for s in lines.values():
        k = len(s)
        if k > maxc:
            maxc = k
        if k == 4:
            count += 1
    return count, maxc


def _valid(points):
    return len(set(map(tuple, points))) == 16 and all(
        abs(x) <= 1000 and abs(y) <= 1000 for x, y in points
    )


def _neighbors(points, rng):
    """Yield (idx, new_point) candidates: random small shifts + random relocations."""
    n = len(points)
    while True:
        idx = rng.randrange(n)
        x, y = points[idx]
        r = rng.random()
        if r < 0.7:
            # small local move
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
        else:
            # larger random relocation
            nx = rng.randint(-1000, 1000)
            ny = rng.randint(-1000, 1000)
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if any(nx == px and ny == py for px, py in points):
            continue
        yield idx, (nx, ny)


def _local_search(points, rng, time_limit):
    """Simulated-annealing-ish local search maximizing 4-lines under no-5 constraint."""
    points = [list(p) for p in points]
    best_points = [list(p) for p in points]
    best_score = _four_lines(points)[0]
    cur_score, _ = _four_lines(points)

    start = time.time()
    T = 2.0
    cooling = 0.99995
    step = 0
    while time.time() < time_limit:
        step += 1
        idx, (nx, ny) = next(_neighbors(points, rng))
        old = points[idx]
        points[idx] = [nx, ny]
        sc, mc = _four_lines(points)
        if mc > 4:
            # invalid: reject
            points[idx] = old
            continue
        delta = sc - cur_score
        if delta > 0 or rng.random() < (T ** (1.0 / max(1, delta)) if delta < 0 else 1.0):
            # accept
            cur_score = sc
            if sc > best_score:
                best_score = sc
                best_points = [list(p) for p in points]
        else:
            points[idx] = old
        # cool down
        T *= cooling
        if T < 0.05:
            T = 0.05
    return best_points, best_score


def construct():
    start = time.time()
    time_limit = start + 150.0 - DEADLINE_MARGIN

    # Seed: the 4x4 grid is a known valid configuration with many 4-point lines.
    base = [[x, y] for x in range(4) for y in range(4)]
    best_points = [list(p) for p in base]
    best_score = _four_lines(base)[0]

    # Try multiple diverse restarts / seeds, each doing a bounded local search.
    rng = random.Random(12345)

    # A few alternative starting configurations to escape the grid basin.
    def make_jittered(seed_val):
        r = random.Random(seed_val)
        pts = [list(p) for p in base]
        for _ in range(16):
            i = r.randrange(16)
            dx, dy = r.randint(-2, 2), r.randint(-2, 2)
            if dx or dy:
                pts[i] = [pts[i][0] + dx, pts[i][1] + dy]
        return pts

    restarts = [base]
    for s in range(12):
        restarts.append(make_jittered(1000 + s))

    for ri, start_pts in enumerate(restarts):
        if time.time() >= time_limit:
            break
        # per-restart budget: split remaining time
        remaining = time_limit - time.time()
        per = max(1.0, remaining / max(1, len(restarts) - ri))
        cand, sc = _local_search(start_pts, rng, time.time() + per)
        if sc > best_score:
            best_score = sc
            best_points = [list(p) for p in cand]

    # Final safety: ensure validity; if not, fall back to the base grid.
    if not _valid(best_points) or _four_lines(best_points)[1] > 4:
        best_points = [list(p) for p in base]

    return [[int(x), int(y)] for x, y in best_points]
