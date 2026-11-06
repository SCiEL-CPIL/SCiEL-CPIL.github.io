"""Improved program: faster incremental scoring + structured grid perturbations + multi-restart search."""
import time
import math
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

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
    a = dy
    b = -dx
    c = a * x1 + b * y1
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _score(points):
    """Count lines containing exactly 4 of the 16 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Build line counts using pairs
    line_count = defaultdict(int)
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(points[i], points[j])
            if k is not None:
                line_count[k] += 1
                line_points[k].add(i)
                line_points[k].add(j)
    
    # For each line, determine how many of the 16 points lie on it
    score = 0
    for k, pair_count in line_count.items():
        # A line through points with pair_count pairs can have at most
        # m points where m*(m-1)/2 >= pair_count, and m*(m-1)/2 < (m+1)*m/2
        # Actually we need to count actual points on the line
        pts_on_line = line_points[k]
        m = len(pts_on_line)
        if m >= 5:
            return -1
        if m == 4:
            score += 1
    
    return score


def _score_fast(points):
    """Faster score using NumPy for the final count check."""
    n = len(points)
    if n != 16:
        return -1
    pts = np.array(points, dtype=np.int64)
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Collect unique lines
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(points[i], points[j])
            if k is not None:
                line_set.add(k)
    
    if not line_set:
        return 0
    
    keys_list = list(line_set)
    a_arr = np.array([k[0] for k in keys_list], dtype=np.int64)
    b_arr = np.array([k[1] for k in keys_list], dtype=np.int64)
    c_arr = np.array([k[2] for k in keys_list], dtype=np.int64)
    
    lhs = a_arr[:, None] * xs[None, :] + b_arr[:, None] * ys[None, :]
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
    return _score(points) >= 0


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_grid_shifted(sx, sy):
    return [[x + sx, y + sy] for x in range(4) for y in range(4)]


def _make_random_points(rng, scale=12):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _perturb_structure(base, rng, n_perturb=4, radius=3):
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


def _try_move_point(points, idx, nx, ny):
    """Try moving point idx to (nx, ny). Return new points if valid, else None."""
    if abs(nx) > 1000 or abs(ny) > 1000:
        return None
    for i in range(16):
        if i != idx and points[i][0] == nx and points[i][1] == ny:
            return None
    new_pts = [list(p) for p in points]
    new_pts[idx] = [nx, ny]
    return new_pts


def _local_search(points, deadline, rng, max_iter=50000):
    """Hill-climb with early stopping."""
    best = [list(p) for p in points]
    best_score = _score_fast(best)
    if best_score < 0:
        best_score = 0
    seen_positions = set()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        order = list(range(16))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            x, y = best[idx]
            # Try nearby positions in increasing radius
            for r in range(1, 6):
                found = False
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) + abs(dy) != r:
                            continue
                        nx, ny = x + dx, y + dy
                        new_pts = _try_move_point(best, idx, nx, ny)
                        if new_pts is None:
                            continue
                        s = _score_fast(new_pts)
                        if s > best_score:
                            best = new_pts
                            best_score = s
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if improved:
                break
        if not improved:
            # Try random jumps
            any_jump = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                idx = rng.randint(0, 15)
                x, y = best[idx]
                nx = x + rng.randint(-20, 20)
                ny = y + rng.randint(-20, 20)
                new_pts = _try_move_point(best, idx, nx, ny)
                if new_pts is None:
                    continue
                s = _score_fast(new_pts)
                if s > best_score:
                    best = new_pts
                    best_score = s
                    any_jump = True
                    break
            if not any_jump:
                break
    
    return best, best_score


def _anneal(points, deadline, rng):
    """Simulated annealing."""
    cur = [list(p) for p in points]
    cur_score = _score_fast(cur)
    if cur_score < 0:
        cur_score = 0
    best = [list(p) for p in cur]
    best_score = cur_score
    T = 3.0
    step = 0
    while time.time() < deadline:
        step += 1
        if step % 300 == 0:
            T *= 0.95
        idx = rng.randint(0, 15)
        x, y = cur[idx]
        dx = rng.randint(-4, 4)
        dy = rng.randint(-4, 4)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        new_pts = _try_move_point(cur, idx, nx, ny)
        if new_pts is None:
            continue
        s = _score_fast(new_pts)
        if s < 0:
            continue
        if s >= cur_score or rng.random() < math.exp((s - cur_score) / max(T, 1e-6)):
            cur = new_pts
            cur_score = s
            if s > best_score:
                best = [list(p) for p in cur]
                best_score = s
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Base 4x4 grid
    grid = _make_grid_4x4()
    candidates.append(grid)
    
    # Shifted grids
    for sx in [-2, -1, 0, 1, 2]:
        for sy in [-2, -1, 0, 1, 2]:
            candidates.append(_make_grid_shifted(sx, sy))
    
    # Structured perturbations of the grid
    for k in range(8):
        candidates.append(_perturb_structure(grid, rng, n_perturb=k + 1, radius=3))
    
    # More aggressive perturbations
    for k in range(4):
        candidates.append(_perturb_structure(grid, rng, n_perturb=2 * (k + 1), radius=5))
    
    # Random configurations
    for _ in range(6):
        candidates.append(_make_random_points(rng, scale=15))
    
    # Scale-variant grids (non-unit spacing)
    for s in [2, 3]:
        candidates.append([[x * s, y * s] for x in range(4) for y in range(4)])
    
    # Mixed: some grid points, some random
    for _ in range(4):
        pts = [list(p) for p in grid]
        for idx in rng.sample(range(16), 4):
            pts[idx] = [rng.randint(-15, 15), rng.randint(-15, 15)]
        # Remove duplicates
        seen = set()
        unique = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique.append(p)
        while len(unique) < 16:
            x = rng.randint(-15, 15)
            y = rng.randint(-15, 15)
            if (x, y) not in seen:
                seen.add((x, y))
                unique.append([x, y])
        candidates.append(unique[:16])
    
    best_points = None
    best_score = -1
    
    # Score all initial candidates, keep top few
    scored = []
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score_fast(cand)
        scored.append((s, cand))
    
    scored.sort(key=lambda x: -x[0])
    
    for s, cand in scored[:8]:
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in cand]
    
    if best_score < 0 or best_points is None:
        return _make_grid_4x4()
    
    # Multi-restart local search from top candidates
    top_candidates = [cand for s, cand in scored[:5] if s >= 0]
    
    for cand in top_candidates:
        if time.time() > deadline - 10:
            break
        ls_deadline = min(deadline - 5, time.time() + 30)
        lp, ls = _local_search(cand, ls_deadline, rng)
        if ls > best_score:
            best_score = ls
            best_points = lp
    
    # Annealing from best
    remaining = deadline - time.time()
    if remaining > 15:
        anneal_deadline = time.time() + remaining * 0.5
        ap, as_ = _anneal(best_points, anneal_deadline, rng)
        if as_ > best_score:
            best_score = as_
            best_points = ap
    
    # Final local search
    if time.time() < deadline - 3:
        fp, fs = _local_search(best_points, deadline, rng)
        if fs > best_score:
            best_score = fs
            best_points = fp
    
    # Final validation
    if not _is_valid(best_points):
        return _make_grid_4x4()
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
