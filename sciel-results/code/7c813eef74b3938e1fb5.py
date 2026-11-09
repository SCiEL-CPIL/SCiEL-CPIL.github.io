import time
import random
import math

_DEADLINE = 135.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _count_exact4(points):
    """Count lines containing exactly 4 of the given 16 points.
    Returns (count, max_collinear)."""
    n = len(points)
    # map from normalized direction+point to set of point indices
    from collections import defaultdict
    line_pts = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g:
                dx //= g
                dy //= g
            # canonicalize direction (make first non-zero positive)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # line key: direction + a point on the line (use the cross product offset)
            # offset = x*dy - y*dx (constant for the line)
            off = xi * dy - yi * dx
            key = (dx, dy, off)
            line_pts[key].add(i)
            line_pts[key].add(j)
    count4 = 0
    maxc = 2
    for key, idxs in line_pts.items():
        c = len(idxs)
        if c > maxc:
            maxc = c
        if c == 4:
            count4 += 1
    return count4, maxc

def _score(points):
    """Objective: maximize exact-4 lines, penalize any 5+ collinear heavily."""
    c4, maxc = _count_exact4(points)
    if maxc >= 5:
        return c4 - 1000  # strong penalty
    return c4

def _random_points(rng):
    return [[rng.randint(-1000, 1000), rng.randint(-1000, 1000)] for _ in range(16)]

def construct():
    rng = random.Random(12345)
    best_pts = None
    best_score = -1

    # Seed with a known good 4x4 grid (scaled to stay in bounds)
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_pts = [list(p) for p in grid]
    best_score = _score(best_pts)

    # Also try a few structured seeds
    structured_seeds = [
        # 4x4 grid
        [[x, y] for x in range(4) for y in range(4)],
        # 4x4 grid offset
        [[x + 1, y + 1] for x in range(4) for y in range(4)],
    ]
    for seed in structured_seeds:
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_pts = [list(p) for p in seed]

    # Simulated annealing
    T = 50.0
    cooling = 0.9999
    T_min = 0.1
    cur_pts = [list(p) for p in best_pts]
    cur_score = best_score

    step = 0
    while _elapsed() < _DEADLINE and T > T_min:
        step += 1
        # Move: perturb one point
        idx = rng.randrange(16)
        ox, oy = cur_pts[idx]
        # small random move
        nx = ox + rng.randint(-15, 15)
        ny = oy + rng.randint(-15, 15)
        # clamp to bounds
        nx = max(-1000, min(1000, nx))
        ny = max(-1000, min(1000, ny))
        # ensure distinct
        if any(cur_pts[k][0] == nx and cur_pts[k][1] == ny for k in range(16) if k != idx):
            continue
        old = cur_pts[idx][:]
        cur_pts[idx] = [nx, ny]
        new_score = _score(cur_pts)
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = [list(p) for p in cur_pts]
        else:
            cur_pts[idx] = old
        T *= cooling
        # occasionally do a bigger jump to escape local optima
        if step % 200 == 0:
            for _ in range(rng.randint(1, 3)):
                idx2 = rng.randrange(16)
                old2 = cur_pts[idx2][:]
                nx2 = old2[0] + rng.randint(-60, 60)
                ny2 = old2[1] + rng.randint(-60, 60)
                nx2 = max(-1000, min(1000, nx2))
                ny2 = max(-1000, min(1000, ny2))
                if any(cur_pts[k][0] == nx2 and cur_pts[k][1] == ny2 for k in range(16) if k != idx2):
                    cur_pts[idx2] = old2
                    continue
                cur_pts[idx2] = [nx2, ny2]
                s2 = _score(cur_pts)
                if s2 < cur_score - 20:
                    cur_pts[idx2] = old2
                else:
                    cur_score = s2
                    if s2 > best_score:
                        best_score = s2
                        best_pts = [list(p) for p in cur_pts]

    # Final validation
    c4, maxc = _count_exact4(best_pts)
    if maxc >= 5:
        # fallback to grid
        best_pts = [[x, y] for x in range(4) for y in range(4)]

    return best_pts
