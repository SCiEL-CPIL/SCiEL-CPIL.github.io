"""Structural rewrite: line-completion heuristic + incremental scoring + multi-restart."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _canonical_line_key(x1, y1, x2, y2):
    """Canonical (a,b,c) for line through two points: a*x + b*y = c, gcd=1, a>0 or (a==0,b>0)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    a = dy // g
    b = -dx // g
    c = a * x1 + b * y1
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _score(points):
    """Count lines with exactly 4 points. Returns -1 if any line has >=5."""
    n = 16
    pts = np.array(points, dtype=np.int64)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Build unique lines
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(xs[i], ys[i], xs[j], ys[j])
            if k is not None:
                line_set.add(k)

    if not line_set:
        return 0

    keys = list(line_set)
    a_arr = np.array([k[0] for k in keys], dtype=np.int64)
    b_arr = np.array([k[1] for k in keys], dtype=np.int64)
    c_arr = np.array([k[2] for k in keys], dtype=np.int64)

    lhs = a_arr[:, None] * xs[None, :] + b_arr[:, None] * ys[None, :]
    counts = np.sum(lhs == c_arr[:, None], axis=1)

    if np.any(counts >= 5):
        return -1
    return int(np.sum(counts == 4))


def _is_valid(points):
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(int(x)) > 1000 or abs(int(y)) > 1000:
            return False
        t = (int(x), int(y))
        if t in seen:
            return False
        seen.add(t)
    return _score(points) >= 0


def _score_and_lines(points):
    """Return (score, dict of line_key -> set of point indices on that line).
    Returns (-1, None) if invalid (5+ collinear)."""
    n = 16
    pts = points  # list of [x,y]
    xs = np.array([p[0] for p in pts], dtype=np.int64)
    ys = np.array([p[1] for p in pts], dtype=np.int64)

    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(xs[i], ys[i], xs[j], ys[j])
            if k is not None:
                line_map[k].add(i)
                line_map[k].add(j)

    # Count points per line
    score = 0
    line_points = {}
    for k, idx_set in line_map.items():
        cnt = len(idx_set)
        if cnt >= 5:
            return -1, None
        line_points[k] = idx_set
        if cnt == 4:
            score += 1
    return score, line_points


def _find_completion_candidates(points, line_points, rng, max_candidates=40):
    """Find candidate (point_index, new_x, new_y) that would complete a 3-point line to 4.
    Only consider lines with exactly 3 of our points; find integer points on that line
    that are not already in our set and within bounds."""
    candidates = []
    # Collect all 3-point lines
    three_lines = [(k, idx_set) for k, idx_set in line_points.items() if len(idx_set) == 3]
    if not three_lines:
        return candidates

    rng.shuffle(three_lines)
    existing_set = set((p[0], p[1]) for p in points)

    # Also collect 2-point lines that are part of potential 4-line structures
    # For each 3-point line, find the 4th point that would complete it
    for k, idx_set in three_lines:
        if len(candidates) >= max_candidates:
            break
        a, b, c = k
        # Get the 3 points
        pts_on_line = [points[i] for i in idx_set]
        
        # Strategy 1: Try to find a 4th point on this line
        # The line is a*x + b*y = c. Find integer solutions.
        # If gcd(a,b) doesn't divide c, no integer solutions (but we have 3, so this won't happen)
        g = math.gcd(abs(a), abs(b))
        
        # Parametrize the line. Find direction vector.
        if a != 0:
            # x = (c - b*y) / a
            # For integer x, need a | (c - b*y)
            # Try y values in a reasonable range around the existing points
            y_min = min(p[1] for p in pts_on_line) - 20
            y_max = max(p[1] for p in pts_on_line) + 20
            tried = set()
            for ny in range(max(-1000, y_min), min(1001, y_max + 1)):
                num = c - b * ny
                if num % a != 0:
                    continue
                nx = num // a
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in existing_set or (nx, ny) in tried:
                    continue
                # Check if this point is "close" to the cluster (within ~30 units)
                if any(abs(nx - p[0]) <= 30 and abs(ny - p[1]) <= 30 for p in pts_on_line):
                    tried.add((nx, ny))
                    candidates.append((nx, ny, k))
                    if len(candidates) >= max_candidates:
                        break
        elif b != 0:
            # b*y = c => y = c/b (fixed)
            if c % b != 0:
                continue
            ny = c // b
            if abs(ny) > 1000:
                continue
            x_min = min(p[0] for p in pts_on_line) - 20
            x_max = max(p[0] for p in pts_on_line) + 20
            tried = set()
            for nx in range(max(-1000, x_min), min(1001, x_max + 1)):
                if (nx, ny) in existing_set or (nx, ny) in tried:
                    continue
                if any(abs(nx - p[0]) <= 30 and abs(ny - p[1]) <= 30 for p in pts_on_line):
                    tried.add((nx, ny))
                    candidates.append((nx, ny, k))
                    if len(candidates) >= max_candidates:
                        break
        else:
            continue
    return candidates


def _local_search_heuristic(points, deadline, rng, max_iter=50000):
    """Heuristic local search: 60% line-completion moves, 25% structured perturbation, 15% random."""
    best = [list(p) for p in points]
    best_score, line_points = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
        line_points = None

    existing_set = set((p[0], p[1]) for p in best)
    no_improve = 0
    max_no_improve = 500

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break

        moved = False
        r = rng.random()
        
        if line_points is not None and r < 0.60:
            # Line-completion move
            cands = _find_completion_candidates(best, line_points, rng, max_candidates=20)
            if cands:
                rng.shuffle(cands)
                for nx, ny, line_k in cands:
                    if (nx, ny) in existing_set:
                        continue
                    # Try moving each point to (nx, ny)
                    # Prefer moving points that are "least useful" (on fewer 3-point lines)
                    point_line_counts = defaultdict(int)
                    for k, idx_set in line_points.items():
                        if len(idx_set) == 3:
                            for idx in idx_set:
                                point_line_counts[idx] += 1
                    
                    # Sort points by usefulness (fewest 3-line memberships first)
                    sorted_indices = sorted(range(16), key=lambda i: point_line_counts.get(i, 0))
                    for idx in sorted_indices:
                        if best[idx][0] == nx and best[idx][1] == ny:
                            continue
                        old_x, old_y = best[idx]
                        new_pts = [list(p) for p in best]
                        new_pts[idx] = [nx, ny]
                        s, lp = _score_and_lines(new_pts)
                        if s > best_score:
                            existing_set.discard((old_x, old_y))
                            existing_set.add((nx, ny))
                            best = new_pts
                            best_score = s
                            line_points = lp
                            no_improve = 0
                            moved = True
                            break
                    if moved:
                        break
        elif r < 0.85:
            # Structured perturbation: move a point along a line direction
            if line_points is not None:
                # Pick a random 3-point line
                three_lines = [(k, idx_set) for k, idx_set in line_points.items() if len(idx_set) == 3]
                if three_lines:
                    k, idx_set = rng.choice(three_lines)
                    a, b, c = k
                    # Pick a point on this line to move
                    idx = rng.choice(list(idx_set))
                    x, y = best[idx]
                    # Move along the line direction
                    # Direction vector: (b, -a) or (-b, a)
                    dx_dir, dy_dir = b, -a
                    g = math.gcd(abs(dx_dir), abs(dy_dir))
                    if g > 0:
                        dx_dir //= g
                        dy_dir //= g
                    # Try several steps
                    for step in [1, -1, 2, -2, 3, -3]:
                        nx = x + step * dx_dir
                        ny = y + step * dy_dir
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        if (nx, ny) in existing_set:
                            continue
                        new_pts = [list(p) for p in best]
                        new_pts[idx] = [nx, ny]
                        s, lp = _score_and_lines(new_pts)
                        if s > best_score:
                            existing_set.discard((x, y))
                            existing_set.add((nx, ny))
                            best = new_pts
                            best_score = s
                            line_points = lp
                            no_improve = 0
                            moved = True
                            break
        else:
            # Random perturbation with larger steps
            idx = rng.randint(0, 15)
            x, y = best[idx]
            # Try a few random offsets with larger range
            for _try in range(10):
                dx = rng.randint(-6, 6)
                dy = rng.randint(-6, 6)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in existing_set:
                    continue
                new_pts = [list(p) for p in best]
                new_pts[idx] = [nx, ny]
                s, lp = _score_and_lines(new_pts)
                if s > best_score:
                    existing_set.discard((x, y))
                    existing_set.add((nx, ny))
                    best = new_pts
                    best_score = s
                    line_points = lp
                    no_improve = 0
                    moved = True
                    break
        if not moved:
            no_improve += 1

    return best, best_score


def _anneal(points, deadline, rng):
    """Simulated annealing with line-completion bias and structured moves."""
    cur = [list(p) for p in points]
    cur_score, cur_lp = _score_and_lines(cur)
    if cur_score < 0:
        cur_score = 0
        cur_lp = None
    best = [list(p) for p in cur]
    best_score = cur_score
    existing_set = set((p[0], p[1]) for p in cur)
    T = 5.0
    step = 0
    no_improve = 0
    max_no_improve = 1000

    while time.time() < deadline:
        step += 1
        if step % 300 == 0:
            T *= 0.95
            if T < 0.01:
                break
        if no_improve > max_no_improve:
            break

        r = rng.random()
        
        if cur_lp is not None and r < 0.50:
            # Line-completion move
            cands = _find_completion_candidates(cur, cur_lp, rng, max_candidates=10)
            if not cands:
                # Fall through to random
                idx = rng.randint(0, 15)
                x, y = cur[idx]
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
            else:
                nx, ny, _ = rng.choice(cands)
                # Pick a point to move - prefer less useful points
                point_line_counts = defaultdict(int)
                for k, idx_set in cur_lp.items():
                    if len(idx_set) == 3:
                        for idx in idx_set:
                            point_line_counts[idx] += 1
                sorted_indices = sorted(range(16), key=lambda i: point_line_counts.get(i, 0))
                idx = sorted_indices[rng.randint(0, min(8, 15))]
        elif r < 0.80 and cur_lp is not None:
            # Structured move along a line
            three_lines = [(k, idx_set) for k, idx_set in cur_lp.items() if len(idx_set) == 3]
            if three_lines:
                k, idx_set = rng.choice(three_lines)
                a, b, c = k
                idx = rng.choice(list(idx_set))
                x, y = cur[idx]
                dx_dir, dy_dir = b, -a
                g = math.gcd(abs(dx_dir), abs(dy_dir))
                if g > 0:
                    dx_dir //= g
                    dy_dir //= g
                step_size = rng.choice([1, -1, 2, -2])
                nx = x + step_size * dx_dir
                ny = y + step_size * dy_dir
            else:
                idx = rng.randint(0, 15)
                x, y = cur[idx]
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
        else:
            # Random move
            idx = rng.randint(0, 15)
            x, y = cur[idx]
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy

        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in existing_set:
            continue

        new_pts = [list(p) for p in cur]
        new_pts[idx] = [nx, ny]
        s, lp = _score_and_lines(new_pts)
        if s < 0:
            no_improve += 1
            continue
        
        delta = s - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-6)):
            existing_set.discard((cur[idx][0], cur[idx][1]))
            existing_set.add((nx, ny))
            cur = new_pts
            cur_score = s
            cur_lp = lp
            if s > best_score:
                best = [list(p) for p in cur]
                best_score = s
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

    return best, best_score


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_perturbed_grid(rng, n_perturb, radius):
    grid = _make_grid_4x4()
    pts = [list(p) for p in grid]
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


def _make_random_points(rng, scale=6):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    rng = random.Random(42)

    # Generate diverse initial candidates
    candidates = []
    candidates.append(_make_grid_4x4())
    for k in range(1, 6):
        candidates.append(_make_perturbed_grid(rng, k, 3))
    for _ in range(5):
        candidates.append(_make_random_points(rng, scale=10))
    for _ in range(3):
        candidates.append(_make_random_points(rng, scale=5))

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
        return _make_grid_4x4()

    # Multi-restart local search with heuristic
    remaining = deadline - time.time()
    n_restarts = max(3, int(remaining / 20))  # ~20s per restart
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        restart_deadline = time.time() + min(18, deadline - time.time() - 3)

        # For restarts after first, create diverse starting points
        if r == 0:
            start_pts = best_points
        else:
            # Mix of strategies: perturb best, new grid, random
            strat = rng.random()
            if strat < 0.4:
                # Perturb the best solution
                start_pts = [list(p) for p in best_points]
                for _ in range(rng.randint(2, 6)):
                    idx = rng.randint(0, 15)
                    dx = rng.randint(-4, 4)
                    dy = rng.randint(-4, 4)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = start_pts[idx][0] + dx, start_pts[idx][1] + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        if (nx, ny) not in [(p[0], p[1]) for i, p in enumerate(start_pts) if i != idx]:
                            start_pts[idx] = [nx, ny]
            elif strat < 0.7:
                # New perturbed grid
                start_pts = _make_perturbed_grid(rng, rng.randint(1, 5), 4)
            else:
                # Random points
                start_pts = _make_random_points(rng, scale=12)

        pts, s = _local_search_heuristic(start_pts, restart_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts

    # Final annealing with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        anneal_deadline = time.time() + remaining * 0.65
        pts, s = _anneal(best_points, anneal_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts

        # Final local search polish
        if time.time() < deadline - 3:
            pts, s = _local_search_heuristic(best_points, deadline - 2, rng, max_iter=30000)
            if s > best_score:
                best_score = s
                best_points = pts

    # Final validation
    if not _is_valid(best_points):
        return _make_grid_4x4()

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
