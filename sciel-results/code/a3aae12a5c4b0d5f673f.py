"""Combined program: fast NumPy scorer + aggressive local search + annealing."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _canonical_line_key(p1, p2):
    """Canonical integer key for the line through two integer points.
    Returns (a, b, c) for ax + by = c with gcd(a,b)=1 and a>0 (or a==0,b>0).
    Returns None for identical points.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # a = dy, b = -dx  (line: dy*x - dx*y = dy*x1 - dx*y1)
    a = dy
    b = -dx
    c = a * x1 + b * y1
    # Canonical: make (a,b) have gcd 1 (already), and leading positive
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _score(points):
    """Count lines containing exactly 4 of the 16 points.
    Returns -1 if any line has >= 5 points (invalid).
    Uses NumPy for speed: for each unique line, vectorized count.
    """
    n = len(points)
    if n != 16:
        return -1
    pts = np.array(points, dtype=np.int64)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Build unique lines from all pairs
    line_keys = {}  # key -> (a, b, c)
    keys_list = []
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            k = _canonical_line_key((x1, y1), points[j])
            if k is None:
                continue
            if k not in line_keys:
                line_keys[k] = k
                keys_list.append(k)

    if not keys_list:
        return 0

    # Vectorized: for each line (a,b,c), count points with a*x + b*y == c
    # Stack all lines into arrays
    a_arr = np.array([k[0] for k in keys_list], dtype=np.int64)
    b_arr = np.array([k[1] for k in keys_list], dtype=np.int64)
    c_arr = np.array([k[2] for k in keys_list], dtype=np.int64)

    # lhs[i, j] = a_i * x_j + b_i * y_j
    lhs = a_arr[:, None] * xs[None, :] + b_arr[:, None] * ys[None, :]
    # counts[i] = number of points on line i
    counts = np.sum(lhs == c_arr[:, None], axis=1)

    if np.any(counts >= 5):
        return -1

    return int(np.sum(counts == 4))


def _is_valid(points):
    """Check 16 distinct integer points, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            return False
        if abs(int(x)) > 1000 or abs(int(y)) > 1000:
            return False
        t = (int(x), int(y))
        if t in seen:
            return False
        seen.add(t)
    # No 5 collinear (use score which returns -1 on violation)
    return _score(points) >= 0


def _local_search(points, deadline, rng, max_iter=None):
    """Hill-climb: try moving one point at a time to nearby integer positions.
    Accept only if score strictly improves. Returns (best_points, best_score).
    """
    best = [list(p) for p in points]
    best_score = _score(best)
    if best_score < 0:
        best_score = 0

    if max_iter is None:
        max_iter = 100000

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        # Try each point in random order
        order = list(range(16))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            x, y = best[idx]
            # Try a ring of nearby positions, radius 1..4
            for r in range(1, 5):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) + abs(dy) > r:  # only the "outer ring"
                            continue
                        nx, ny = x + dx, y + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        # Quick duplicate check
                        if (nx, ny) in [(p[0], p[1]) for i, p in enumerate(best) if i != idx]:
                            continue
                        new_pts = [list(p) for p in best]
                        new_pts[idx] = [nx, ny]
                        s = _score(new_pts)
                        if s > best_score:
                            best = new_pts
                            best_score = s
                            x, y = nx, ny
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try a few larger random jumps
            for _ in range(30):
                if time.time() > deadline:
                    break
                idx = rng.randint(0, 15)
                x, y = best[idx]
                nx = x + rng.randint(-30, 30)
                ny = y + rng.randint(-30, 30)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in [(p[0], p[1]) for i, p in enumerate(best) if i != idx]:
                    continue
                new_pts = [list(p) for p in best]
                new_pts[idx] = [nx, ny]
                s = _score(new_pts)
                if s > best_score:
                    best = new_pts
                    best_score = s
                    break
            else:
                # Stuck; stop local search
                break

    return best, best_score


def _anneal(points, deadline, rng):
    """Simulated annealing: accept worse moves with probability exp(-d/T)."""
    cur = [list(p) for p in points]
    cur_score = _score(cur)
    if cur_score < 0:
        cur_score = 0
    best = [list(p) for p in cur]
    best_score = cur_score
    T = 2.0
    step = 0
    while time.time() < deadline:
        step += 1
        # Cool down
        if step % 500 == 0:
            T *= 0.97
        idx = rng.randint(0, 15)
        x, y = cur[idx]
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in [(p[0], p[1]) for i, p in enumerate(cur) if i != idx]:
            continue
        new_pts = [list(p) for p in cur]
        new_pts[idx] = [nx, ny]
        s = _score(new_pts)
        if s < 0:
            continue  # invalid, reject
        if s >= cur_score or rng.random() < math.exp((s - cur_score) / max(T, 1e-6)):
            cur = new_pts
            cur_score = s
            if s > best_score:
                best = [list(p) for p in cur]
                best_score = s
    return best, best_score


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_random_points(rng, scale=8):
    """Generate 16 random distinct integer points within bounds."""
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _perturb_structure(base, rng, n_perturb=4, radius=2):
    """Perturb n_perturb random points of base by small offsets."""
    pts = [list(p) for p in base]
    for _ in range(n_perturb):
        idx = rng.randint(0, 15)
        dx = rng.randint(-radius, radius)
        dy = rng.randint(-radius, radius)
        if dx == 0 and dy == 0:
            continue
        nx, ny = pts[idx][0] + dx, pts[idx][1] + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in [(p[0], p[1]) for i, p in enumerate(pts) if i != idx]:
            continue
        pts[idx] = [nx, ny]
    return pts


def construct():
    start_time = time.time()
    # Evaluator kills at 150s; use 120s deadline for safety margin
    deadline = start_time + 120

    rng = random.Random(12345)

    # Initial candidates: 4x4 grid (known 20 four-point lines) + a few perturbations
    candidates = []
    grid = _make_grid_4x4()
    candidates.append(grid)

    # A few random perturbations of the grid
    for k in range(6):
        candidates.append(_perturb_structure(grid, rng, n_perturb=k + 1, radius=2))

    # A couple of random configurations
    for _ in range(4):
        candidates.append(_make_random_points(rng, scale=10))

    best_points = None
    best_score = -1

    # Score all initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in cand]

    if best_score < 0:
        # Fallback: return the grid (it's always valid)
        return _make_grid_4x4()

    # Phase 1: Local search from best
    best_points, best_score = _local_search(best_points, deadline, rng)

    # Phase 2: Annealing from best (use remaining time)
    remaining = deadline - time.time()
    if remaining > 10:
        anneal_deadline = time.time() + remaining * 0.6
        best_points, s2 = _anneal(best_points, anneal_deadline, rng)
        if s2 > best_score:
            best_score = s2

    # Phase 3: More local search from the annealed result
    if time.time() < deadline - 5:
        best_points, s3 = _local_search(best_points, deadline, rng)
        if s3 > best_score:
            best_score = s3

    # Final validation
    if not _is_valid(best_points):
        # Fallback to grid
        return _make_grid_4x4()

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
