"""Combined: affine-plane-inspired init + incremental scoring + multi-restart + line-completion polish."""
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
    pts = points
    xs = np.array([p[0] for p in pts], dtype=np.int64)
    ys = np.array([p[1] for p in pts], dtype=np.int64)

    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(xs[i], ys[i], xs[j], ys[j])
            if k is not None:
                line_map[k].add(i)
                line_map[k].add(j)

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


def _score_and_lines_fast(points, old_points, changed_idx, old_line_points):
    """Incremental scoring: only recompute lines involving the changed point.
    Returns (score, line_points) or (-1, None) if invalid."""
    n = 16
    if old_line_points is None:
        return _score_and_lines(points)

    new_pts = points
    old_pts = old_points
    cx, cy = new_pts[changed_idx]
    ox, oy = old_pts[changed_idx]

    # Lines that were affected: all lines through the old point position
    # and all lines through the new point position
    affected_keys = set()

    # Find lines that contained the old point
    for k, idx_set in old_line_points.items():
        if changed_idx in idx_set:
            affected_keys.add(k)

    # Find lines through the new point with each other point
    for j in range(n):
        if j == changed_idx:
            continue
        k = _canonical_line_key(cx, cy, new_pts[j][0], new_pts[j][1])
        if k is not None:
            affected_keys.add(k)

    # Rebuild line_map from scratch but only need to check affected lines
    # Actually, let's just rebuild fully but use the fast path for checking
    # For 16 points, full rebuild is ~120 key computations which is fast enough
    # The real win is avoiding numpy array construction
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _canonical_line_key(new_pts[i][0], new_pts[i][1], new_pts[j][0], new_pts[j][1])
            if k is not None:
                line_map[k].add(i)
                line_map[k].add(j)

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
    """Find candidate (point_index, new_x, new_y) that would complete a 3-point line to 4."""
    candidates = []
    three_lines = [(k, idx_set) for k, idx_set in line_points.items() if len(idx_set) == 3]
    if not three_lines:
        return candidates

    rng.shuffle(three_lines)
    existing_set = set((p[0], p[1]) for p in points)

    for k, idx_set in three_lines:
        if len(candidates) >= max_candidates:
            break
        a, b, c = k
        pts_on_line = [points[i] for i in idx_set]
        tried = set()
        for px, py in pts_on_line:
            for dy in range(-6, 7):
                ny = py + dy
                if abs(ny) > 1000:
                    continue
                if a != 0:
                    num = c - b * ny
                    if num % a != 0:
                        continue
                    nx = num // a
                else:
                    if b * ny != c:
                        continue
                    for dx in range(-6, 7):
                        nx2 = px + dx
                        if abs(nx2) > 1000:
                            continue
                        if (nx2, ny) in existing_set or (nx2, ny) in tried:
                            continue
                        tried.add((nx2, ny))
                        candidates.append((nx2, ny, k))
                    continue
                if abs(nx) > 1000:
                    continue
                if (nx, ny) in existing_set or (nx, ny) in tried:
                    continue
                tried.add((nx, ny))
                candidates.append((nx, ny, k))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _local_search(points, deadline, rng, max_iter=50000):
    """Local search: 70% line-completion moves, 30% random perturbation."""
    best = [list(p) for p in points]
    best_score, line_points = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
        line_points = None

    existing_set = set((p[0], p[1]) for p in best)
    no_improve = 0
    max_no_improve = 400

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break

        moved = False
        if line_points is not None and rng.random() < 0.70:
            cands = _find_completion_candidates(best, line_points, rng, max_candidates=20)
            rng.shuffle(cands)
            for nx, ny, line_k in cands:
                if (nx, ny) in existing_set:
                    continue
                for idx in range(16):
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
        else:
            idx = rng.randint(0, 15)
            x, y = best[idx]
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
    """Simulated annealing with line-completion bias."""
    cur = [list(p) for p in points]
    cur_score, cur_lp = _score_and_lines(cur)
    if cur_score < 0:
        cur_score = 0
        cur_lp = None
    best = [list(p) for p in cur]
    best_score = cur_score
    existing_set = set((p[0], p[1]) for p in cur)
    T = 4.0
    step = 0

    while time.time() < deadline:
        step += 1
        if step % 300 == 0:
            T *= 0.95
            if T < 0.01:
                break

        if cur_lp is not None and rng.random() < 0.5:
            cands = _find_completion_candidates(cur, cur_lp, rng, max_candidates=8)
            if not cands:
                idx = rng.randint(0, 15)
                x, y = cur[idx]
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
            else:
                nx, ny, _ = rng.choice(cands)
                idx = rng.randint(0, 15)
        else:
            idx = rng.randint(0, 15)
            x, y = cur[idx]
            dx = rng.randint(-4, 4)
            dy = rng.randint(-4, 4)
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
            continue
        if s >= cur_score or rng.random() < math.exp((s - cur_score) / max(T, 1e-6)):
            existing_set.discard((cur[idx][0], cur[idx][1]))
            existing_set.add((nx, ny))
            cur = new_pts
            cur_score = s
            cur_lp = lp
            if s > best_score:
                best = [list(p) for p in cur]
                best_score = s

    return best, best_score


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_affine_plane_4x4():
    """
    Attempt to construct a configuration inspired by the affine plane of order 4.
    """
    pts = []
    for x in range(4):
        for y in range(4):
            nx = x + y
            ny = y
            pts.append([nx, ny])
    return pts


def _make_modular_pattern():
    """
    Use a modular arithmetic pattern to create many 4-point lines.
    """
    pts = []
    for i in range(4):
        for j in range(4):
            x = i * 3 + j
            y = i + j * 3
            pts.append([x, y])
    return pts


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


def _make_random_points(rng, scale=8):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _make_sheared_grid(rng, shear):
    """Create a sheared 4x4 grid."""
    pts = []
    for x in range(4):
        for y in range(4):
            nx = x + shear * y
            ny = y
            pts.append([nx, ny])
    return pts


def _make_diagonal_ensemble(rng):
    """Create points on multiple diagonal lines."""
    pts = []
    seen = set()
    slopes = [0, 1, -1, 2, -2, 3, -3]
    for slope in slopes:
        for intercept in range(-2, 3):
            if len(pts) >= 16:
                break
            for x in range(-3, 4):
                y = slope * x + intercept
                if abs(x) <= 1000 and abs(y) <= 1000:
                    if (x, y) not in seen:
                        seen.add((x, y))
                        pts.append([x, y])
                        break
            if len(pts) >= 16:
                break
    while len(pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _polish_exhaustive(points, deadline, rng):
    """High-precision polish: exhaustive single-point moves with wider radius."""
    best = [list(p) for p in points]
    best_score, line_points = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
        line_points = None
    existing_set = set((p[0], p[1]) for p in best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                break
            x, y = best[idx]
            for dx in range(-6, 7):
                for dy in range(-6, 7):
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
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s kill

    rng = random.Random(42)

    # Generate diverse initial candidates
    candidates = []
    candidates.append(_make_grid_4x4())
    candidates.append(_make_affine_plane_4x4())
    candidates.append(_make_modular_pattern())
    candidates.append(_make_sheared_grid(rng, 1))
    candidates.append(_make_sheared_grid(rng, 2))
    candidates.append(_make_sheared_grid(rng, 3))
    candidates.append(_make_diagonal_ensemble(rng))
    
    for k in range(1, 5):
        candidates.append(_make_perturbed_grid(rng, k, 3))
    for _ in range(6):
        candidates.append(_make_random_points(rng, scale=12))
    for _ in range(3):
        candidates.append(_make_random_points(rng, scale=6))

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

    # Multi-restart local search
    remaining = deadline - time.time()
    n_restarts = max(4, int(remaining / 18))  # ~18s per restart
    for r in range(n_restarts):
        if time.time() > deadline - 12:
            break
        restart_deadline = time.time() + min(16, deadline - time.time() - 10)

        if r == 0:
            start_pts = best_points
        else:
            # Diverse starting points for later restarts
            rtype = rng.random()
            if rtype < 0.25:
                start_pts = _make_perturbed_grid(rng, rng.randint(1, 5), 4)
            elif rtype < 0.45:
                start_pts = _make_sheared_grid(rng, rng.randint(1, 4))
            elif rtype < 0.65:
                start_pts = _make_diagonal_ensemble(rng)
            elif rtype < 0.80:
                start_pts = _make_modular_pattern()
                for _ in range(rng.randint(2, 5)):
                    idx = rng.randint(0, 15)
                    dx = rng.randint(-3, 3)
                    dy = rng.randint(-3, 3)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = start_pts[idx][0] + dx, start_pts[idx][1] + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        if (nx, ny) not in [(p[0], p[1]) for i, p in enumerate(start_pts) if i != idx]:
                            start_pts[idx] = [nx, ny]
            else:
                start_pts = [list(p) for p in best_points]
                for _ in range(rng.randint(2, 6)):
                    idx = rng.randint(0, 15)
                    dx = rng.randint(-5, 5)
                    dy = rng.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = start_pts[idx][0] + dx, start_pts[idx][1] + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        if (nx, ny) not in [(p[0], p[1]) for i, p in enumerate(start_pts) if i != idx]:
                            start_pts[idx] = [nx, ny]

        pts, s = _local_search(start_pts, restart_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts

    # Final annealing with remaining time
    remaining = deadline - time.time()
    if remaining > 12:
        anneal_deadline = time.time() + remaining * 0.55
        pts, s = _anneal(best_points, anneal_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts

        # One more local search
        if time.time() < deadline - 6:
            pts, s = _local_search(best_points, deadline - 5, rng, max_iter=40000)
            if s > best_score:
                best_score = s
                best_points = pts

    # Final exhaustive single-point polish with wider radius
    if time.time() < deadline - 1:
        pts, s = _polish_exhaustive(best_points, deadline - 1, rng)
        if s > best_score:
            best_score = s
            best_points = pts

    # Final validation
    if not _is_valid(best_points):
        return _make_grid_4x4()

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
