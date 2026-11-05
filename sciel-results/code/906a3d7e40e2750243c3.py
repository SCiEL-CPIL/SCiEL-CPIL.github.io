"""Combined construction for the 12-point, few-ordinary-lines task.

Strategy:
  - Start from multiple structured seeds (grids, line arrangements, two-line seeds).
  - A fast incremental ordinary-line count (delta evaluation) for search moves.
  - Constraint-inspired refinement: given a current configuration, compute a
    "line partition" (which lines have >=3 points) and try to repair the
    configuration by placing points on high-value lines (3+ point lines) or
    intersections of pairs of lines, which directly targets the partition
    structure rather than random perturbation.
  - Simulated annealing with line-extension moves to escape local minima.
  - Aggressive local search for final polish.
  - Deadline guard to return the best configuration found within budget.
"""
import time
import random
import math
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150 s kill


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


def _line_key(x1, y1, x2, y2):
    """Return canonical (A,B,C) for the line through two points."""
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
    return (A, B, C)


def _build_line_map(points):
    """Build a map: line_key -> list of indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key not in line_map:
                line_map[key] = []
            if i not in line_map[key]:
                line_map[key].append(i)
            if j not in line_map[key]:
                line_map[key].append(j)
    return line_map


def _ordinary_from_line_map(line_map):
    """Count ordinary lines from a line map."""
    return sum(1 for v in line_map.values() if len(v) == 1)


def _delta_ordinary(points, idx, new_pt):
    """Compute the change in ordinary-line count when moving point idx to new_pt.
    Returns (new_score, old_score) efficiently by recomputing affected lines.
    For n=12, a full recompute is ~66 line keys; delta is complex, so we
    just do a fast full recompute but with a pre-built structure."""
    new_pts = [p[:] for p in points]
    new_pts[idx] = list(new_pt)
    return _count_ordinary(new_pts)


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


def _make_seed_two_lines():
    """Böröczky-type: 6 points on y=0, 6 points on y=1 with aligned diagonals."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
    ]
    return pts


def _make_seed_perturbed():
    """4x3 grid with one point moved to create extra 3-point lines."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [4, 3],
    ]
    return pts


def _make_seed_hexagon():
    """Regular hexagon-like arrangement with center points."""
    pts = [
        [0, 0], [2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1],
        [1, -1], [0, 2], [0, -2], [2, 2], [-2, 2], [2, -2]
    ]
    seen = set()
    clean = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            clean.append(p[:])
    return clean


def _line_intersection_int(l1, l2):
    """Compute integer intersection of two lines given as (A,B,C).
    Returns (x, y) if it's an integer point, else None."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # parallel or coincident
    x_num = B1 * C2 - B2 * C1
    y_num = C1 * A2 - C2 * A1
    # Note: A x + B y = -C, so x = (B1*C2 - B2*C1)/det, y = (C1*A2 - C2*A1)/det
    if x_num % det != 0 or y_num % det != 0:
        return None
    x = x_num // det
    y = y_num // det
    if abs(x) > 10**9 or abs(y) > 10**9:
        return None
    return (x, y)


def _partition_refine(points, max_iters=50000):
    """Constraint-inspired refinement: use line partition structure to guide moves.
    
    Idea: compute the current line partition. For each point, identify the
    lines it lies on. Try to move points to positions that are:
    1. Intersections of two existing high-count lines (creates new 3+ lines)
    2. On an existing 3+ line (extends it)
    3. Intersections of a high-count line and a 2-count line
    """
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(42)
    iters = 0
    
    while iters < max_iters and time.time() < _DEADLINE - 10:
        iters += 1
        line_map = _build_line_map(cur)
        cur_set = set(tuple(p) for p in cur)
        
        # Identify high-value lines (3+ points) and their intersections
        high_lines = {k: v for k, v in line_map.items() if len(v) >= 3}
        two_lines = {k: v for k, v in line_map.items() if len(v) == 2}
        
        # Generate candidate positions
        candidates = []
        
        # Intersections of pairs of high-value lines
        hl_keys = list(high_lines.keys())
        for i in range(len(hl_keys)):
            for j in range(i + 1, len(hl_keys)):
                pt = _line_intersection_int(hl_keys[i], hl_keys[j])
                if pt is not None and pt not in cur_set:
                    candidates.append(pt)
        
        # Points on high-value lines (extensions)
        for lk, indices in high_lines.items():
            if len(indices) < 2:
                continue
            x1, y1 = cur[indices[0]]
            x2, y2 = cur[indices[1]]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-3, 4):
                for base_idx in indices[:2]:
                    bx, by = cur[base_idx]
                    pt = (bx + k * dx, by + k * dy)
                    if pt not in cur_set and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                        candidates.append(pt)
        
        # Intersections of high lines and 2-point lines
        tl_keys = list(two_lines.keys())
        if tl_keys:
            for hl_k in hl_keys[:5]:  # limit for speed
                for tl_k in tl_keys[:5]:
                    pt = _line_intersection_int(hl_k, tl_k)
                    if pt is not None and pt not in cur_set:
                        candidates.append(pt)
        
        # Random candidates as fallback
        for _ in range(20):
            idx = rng.randrange(12)
            ox, oy = cur[idx]
            pt = (ox + rng.randint(-6, 6), oy + rng.randint(-6, 6))
            if abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                candidates.append(pt)
        
        # Try candidates
        improved = False
        rng.shuffle(candidates)
        for cand in candidates[:30]:
            if cand in cur_set:
                continue
            if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
                continue
            
            # Try moving each point to this candidate
            for idx in range(12):
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
                break
        
        if not improved:
            # No improvement found; do a small random perturbation to escape
            idx = rng.randrange(12)
            ox, oy = cur[idx]
            for _ in range(30):
                pt = (ox + rng.randint(-8, 8), oy + rng.randint(-8, 8))
                if pt in cur_set or abs(pt[0]) > 10**9 or abs(pt[1]) > 10**9:
                    continue
                new_pts = [p[:] for p in cur]
                new_pts[idx] = list(pt)
                if not _valid(new_pts):
                    continue
                new_score = _count_ordinary(new_pts)
                if new_score <= cur_score:
                    cur = new_pts
                    cur_score = new_score
                    cur_set = set(tuple(p) for p in cur)
                    break
    
    return cur, cur_score


def _simulated_annealing(points, max_iters=400000, seed=7):
    """Simulated annealing to escape local minima."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    rng = random.Random(seed)
    T = 2.0
    T_min = 0.01
    cooling = 0.99998
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 10:
        iters += 1
        if T < T_min:
            T = T_min
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)

        # Generate candidate
        if rng.random() < 0.4:
            # Small perturbation
            nx = old[0] + rng.randint(-5, 5)
            ny = old[1] + rng.randint(-5, 5)
            cand = (nx, ny)
        elif rng.random() < 0.8:
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
            k = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
            base = cur[i] if rng.random() < 0.5 else cur[j]
            cand = (base[0] + k * dx, base[1] + k * dy)
        else:
            # Intersection of two random lines
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j:
                j = (i + 1) % 12
            if k == i or k == j:
                k = (i + 2) % 12
            l1 = _line_key(cur[i][0], cur[i][1], cur[j][0], cur[j][1])
            l2 = _line_key(cur[i][0], cur[i][1], cur[k][0], cur[k][1])
            pt = _line_intersection_int(l1, l2)
            if pt is None:
                pt = (old[0] + rng.randint(-5, 5), old[1] + rng.randint(-5, 5))
            cand = pt

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

        if delta < 0 or (rng.random() < math.exp(-delta / T) if delta > 0 else 1.0):
            cur = new_pts
            cur_score = new_score
            cur_set = set(tuple(p) for p in cur)
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]

        T *= cooling

    return best, best_score


def _local_search(points, max_iters=200000, seed=123):
    """Aggressive local search with dynamic candidate generation."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(seed)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 10:
        iters += 1
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        improved = False
        # Strategy 1: small perturbation
        for _ in range(20):
            nx = old[0] + rng.randint(-6, 6)
            ny = old[1] + rng.randint(-6, 6)
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
        for _ in range(15):
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
            k = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
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
    seeds.append(_make_seed_two_lines())
    seeds.append(_make_seed_perturbed())
    seeds.append(_make_seed_hexagon())
    # Scaled grids
    for scale in [2, 3]:
        seeds.append(_make_seed_grid(3, 4, 0, 0, scale))
        seeds.append(_make_seed_grid(4, 3, 0, 0, scale))

    # Evaluate all seeds
    for seed in seeds:
        if len(seed) == 12:
            _consider(seed)

    # Partition refinement from best seed
    if best_pts is not None and time.time() < _DEADLINE - 20:
        pts_pr, score_pr = _partition_refine(best_pts, max_iters=30000)
        if score_pr < best_score:
            best_score = score_pr
            best_pts = pts_pr

    # Simulated annealing from best seed
    if best_pts is not None and time.time() < _DEADLINE - 20:
        pts_sa, score_sa = _simulated_annealing(best_pts, max_iters=400000)
        if score_sa < best_score:
            best_score = score_sa
            best_pts = pts_sa

    # Local search from best so far
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_ls, score_ls = _local_search(best_pts, max_iters=200000)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # Another partition refinement round
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_pr2, score_pr2 = _partition_refine(best_pts, max_iters=20000)
        if score_pr2 < best_score:
            best_score = score_pr2
            best_pts = pts_pr2

    # Another SA round from best
    if best_pts is not None and time.time() < _DEADLINE - 12:
        pts_sa2, score_sa2 = _simulated_annealing(best_pts, max_iters=300000, seed=99)
        if score_sa2 < best_score:
            best_score = score_sa2
            best_pts = pts_sa2

    # Final local search
    if best_pts is not None and time.time() < _DEADLINE - 8:
        pts_final, score_final = _local_search(best_pts, max_iters=150000, seed=555)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    # Final partition refinement
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts_pr3, score_pr3 = _partition_refine(best_pts, max_iters=10000)
        if score_pr3 < best_score:
            best_score = score_pr3
            best_pts = pts_pr3

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
