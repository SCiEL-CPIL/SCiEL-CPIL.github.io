import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_exact4(points):
    """Count lines containing exactly 4 of the points. Fast: 120 pairs only."""
    n = len(points)
    if n < 4:
        return 0
    line_pts = {}
    for i in range(n):
        xi, yi = points[i][0], points[i][1]
        for j in range(i + 1, n):
            xj, yj = points[j][0], points[j][1]
            dx = xj - xi
            dy = yj - yi
            if dx == 0:
                key = (1, 0, xi)
            elif dy == 0:
                key = (0, 1, yi)
            else:
                if dx < 0:
                    dx = -dx
                    dy = -dy
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # line: dy*x - dx*y = c
                c = dy * xi - dx * yi
                key = (dy, -dx, c)
            if key in line_pts:
                line_pts[key].add(j)
            else:
                line_pts[key] = {i, j}
    cnt = 0
    for s in line_pts.values():
        if len(s) == 4:
            cnt += 1
    return cnt

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _valid(points):
    """Check 16 distinct points, |x|,|y|<=1000, no five collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    n = len(points)
    line_pts = {}
    for i in range(n):
        xi, yi = points[i][0], points[i][1]
        for j in range(i + 1, n):
            xj, yj = points[j][0], points[j][1]
            dx = xj - xi
            dy = yj - yi
            if dx == 0:
                key = (1, 0, xi)
            elif dy == 0:
                key = (0, 1, yi)
            else:
                if dx < 0:
                    dx = -dx
                    dy = -dy
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                c = dy * xi - dx * yi
                key = (dy, -dx, c)
            if key in line_pts:
                line_pts[key].add(j)
            else:
                line_pts[key] = {i, j}
    for s in line_pts.values():
        if len(s) >= 5:
            return False
    return True

def _base_points():
    """Deterministic strong base: 4x4 grid with 10 exact-4 lines."""
    return [[x, y] for x in range(4) for y in range(4)]

def _candidate_points(points):
    """Generate candidate integer points: complete 3-collinear sets + neighbors."""
    n = len(points)
    cand = set()
    # Complete 3-collinear sets: for each pair, the third point of any 3-set
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = points[i][0], points[i][1]
            xj, yj = points[j][0], points[j][1]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            # points collinear with i,j: i + k*(dx,dy) for integer k
            for k in range(-6, 7):
                if k == 0 or k == 1:
                    continue
                x = xi + k * dx
                y = yi + k * dy
                if abs(x) <= 1000 and abs(y) <= 1000:
                    cand.add((x, y))
    # Neighbor points
    for x, y in points:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    cand.add((nx, ny))
    return [list(p) for p in cand]

def _local_search(points, deadline):
    """Hill-climb: replace a point with a candidate to improve exact-4 count."""
    best = [p[:] for p in points]
    best_score = _count_exact4(best)
    no_improve = 0
    max_no_improve = 800
    rng = random.Random(12345)
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random point to replace
        idx = rng.randrange(16)
        old = best[idx]
        improved = False
        # Try candidates; prefer those completing 3-collinear
        cand = _candidate_points(best)
        rng.shuffle(cand)
        for new in cand:
            if new == old:
                continue
            # quick reject: would create 5 collinear?
            trial = best[:]
            trial[idx] = new
            # check no 5 collinear (fast)
            if _valid(trial):
                s = _count_exact4(trial)
                if s > best_score:
                    best_score = s
                    best = trial
                    no_improve = 0
                    improved = True
                    break
        if not improved:
            no_improve += 1
    return best, best_score

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    # Start from strong deterministic base
    best = _base_points()
    best_score = _count_exact4(best)

    # Local search from base
    best, best_score = _local_search(best, deadline)

    # Try a few random restarts with perturbation if time remains
    rng = random.Random(777)
    attempts = 0
    while time.time() < deadline - 2 and attempts < 40:
        attempts += 1
        # Perturb: take base, move a few points randomly
        cand = _base_points()
        for _ in range(rng.randint(1, 5)):
            idx = rng.randrange(16)
            nx = rng.randint(-30, 30)
            ny = rng.randint(-30, 30)
            cand[idx] = [nx, ny]
        if not _valid(cand):
            continue
        s = _count_exact4(cand)
        if s >= best_score:
            cand, s = _local_search(cand, deadline)
            if s > best_score:
                best_score = s
                best = cand
        if time.time() > deadline - 5:
            break

    return [[int(x), int(y)] for x, y in best]
# EVOLVE-BLOCK-END
