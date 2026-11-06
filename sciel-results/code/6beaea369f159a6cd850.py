"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key (a,b,c) for the line through p1,p2."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count (full recompute)."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _fast_state(points):
    """Build (line_counts, ordinary_count) for a point set."""
    line_counts = {}
    for i in range(12):
        pi = points[i]
        for j in range(i + 1, 12):
            key = _line_key(pi, points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for v in line_counts.values() if v == 1)
    return line_counts, ordinary

def _move_delta(line_counts, points, i, cand):
    """
    Return (delta_ordinary, new_line_counts) after replacing points[i] with cand.
    If cand collides with another point, return (None, None).
    """
    cand = tuple(cand)
    for j in range(12):
        if j != i and tuple(points[j]) == cand:
            return None, None
    # lines through i (to be removed)
    inc = {}
    for j in range(12):
        if j == i:
            continue
        key = _line_key(points[i], points[j])
        if key is not None:
            inc[key] = inc.get(key, 0) + 1
    # lines through cand (to be added)
    dec = {}
    for j in range(12):
        if j == i:
            continue
        key = _line_key(cand, points[j])
        if key is not None:
            dec[key] = dec.get(key, 0) + 1
    lc = dict(line_counts)
    delta = 0
    for key, cnt in inc.items():
        old = lc.get(key, 0)
        new = old - cnt
        if new == 0:
            del lc[key]
            if old == 1:
                delta -= 1
        else:
            lc[key] = new
            if new == 1:
                delta += 1
    for key, cnt in dec.items():
        old = lc.get(key, 0)
        new = old + cnt
        lc[key] = new
        if new == 1:
            delta -= 1
        elif old == 1:
            delta += 1
    return delta, lc

def _build_hesse_like():
    """
    Deterministic Hesse-like construction: exhaustively remove 4 points from a
    4x4 grid (and a few other structured grids) to maximize 3-point lines.
    This is the structural core of the configuration.
    """
    best_points = None
    best_score = float('inf')

    grids = [
        [(i, j) for i in range(4) for j in range(4)],
        [(i, j) for i in range(3) for j in range(4)],
        [(i, j) for i in range(6) for j in range(2)],
        [(i, j) for i in range(2) for j in range(6)],
        [(i, j) for i in range(12) for j in range(1)],
        [(i, j) for i in range(1) for j in range(12)],
    ]
    for grid in grids:
        if len(grid) == 12:
            if _is_valid(grid):
                s = _count_ordinary_fast(grid)
                if s < best_score:
                    best_score = s
                    best_points = [list(p) for p in grid]
                continue
        # try removing 4 points from this grid
        for removed in combinations(range(len(grid)), 4):
            pts = [grid[k] for k in range(len(grid)) if k not in removed]
            if not _is_valid(pts):
                continue
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
                if best_score <= 6:
                    return best_points, best_score

    # Fallback: 4x4 grid minus 4 corners
    best_points = [[0, 0], [1, 0], [2, 0], [3, 0],
                   [0, 1], [1, 1], [2, 1], [3, 1],
                   [0, 2], [1, 2], [2, 2], [3, 2]]
    best_score = _count_ordinary_fast(best_points)
    return best_points, best_score

def _try_structured_configs():
    """Try various structured line-bundle configurations."""
    configs = []

    # 4+4+4 on three parallel lines with offsets
    for off1 in range(-3, 4):
        for off2 in range(-3, 4):
            cfg = [[x, 0] for x in range(4)] + \
                  [[x + off1, 1] for x in range(4)] + \
                  [[x + off2, 2] for x in range(4)]
            if len(set(tuple(p) for p in cfg)) == 12:
                configs.append(cfg)

    # 4+4+4 on non-parallel lines
    c1 = [[0, 0], [1, 0], [2, 0], [3, 0],
          [1, 1], [2, 2], [3, 3], [4, 4],
          [5, 0], [4, 1], [3, 2], [2, 3]]
    if len(set(tuple(p) for p in c1)) == 12:
        configs.append(c1)

    c2 = [[0, 0], [1, 0], [2, 0], [3, 0],
          [0, 1], [1, 1], [2, 1], [3, 1],
          [4, 0], [5, -1], [6, -2], [7, -3]]
    if len(set(tuple(p) for p in c2)) == 12:
        configs.append(c2)

    best_points = None
    best_score = float('inf')
    for cfg in configs:
        if not _is_valid(cfg):
            continue
        s = _count_ordinary_fast(cfg)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
        if best_score <= 6:
            break
    return best_points, best_score

def _local_search(points, deadline):
    """
    Fast local search using incremental scoring.
    Tries single-point replacement moves from a candidate pool derived from
    lines through existing points plus nearby integer points.
    """
    current = [tuple(p) for p in points]
    line_counts, current_score = _fast_state(current)
    best = [list(p) for p in current]
    best_score = current_score

    # Candidate pool: points on lines through pairs, extended, plus nearby
    coords = set()
    for i in range(12):
        for j in range(i + 1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            for t in range(-4, 5):
                coords.add((x1 + t * dx, y1 + t * dy))
    for p in current:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                coords.add((p[0] + dx, p[1] + dy))
    # Add a broader structured region
    for x in range(-8, 16):
        for y in range(-8, 16):
            coords.add((x, y))

    candidate_list = list(coords)
    random.shuffle(candidate_list)

    no_improve = 0
    max_no_improve = 40
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                delta, new_lc = _move_delta(line_counts, current, i, cand)
                if delta is None:
                    continue
                new_score = current_score + delta
                if new_score < current_score:
                    current = list(current)
                    current[i] = cand
                    current_score = new_score
                    line_counts = new_lc
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        if not improved:
            no_improve += 1
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(12345)

    best_points = None
    best_score = float('inf')

    # 1) Deterministic Hesse-like construction (structural core)
    hp, hs = _build_hesse_like()
    if hs < best_score:
        best_score = hs
        best_points = hp

    # 2) Structured line-bundle configurations
    sp, ss = _try_structured_configs()
    if ss < best_score:
        best_score = ss
        best_points = sp

    # 3) Local search from best deterministic configuration
    if time.time() < deadline - 10:
        ls_points, ls_score = _local_search(best_points, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points

    # 4) Random restarts if still above 6
    if best_score > 6 and time.time() < deadline - 15:
        for restart in range(80):
            if time.time() >= deadline - 10:
                break
            # 3 lines with 4 points each, random slopes/intercepts
            pts = []
            x1s = random.sample(range(-10, 10), 4)
            for x in x1s:
                pts.append((x, 0))
            c2 = random.randint(-5, 5)
            for x in random.sample(range(-10, 10), 4):
                pt = (x, x + c2)
                if all(pt != p for p in pts):
                    pts.append(pt)
                else:
                    for x2 in range(-15, 15):
                        pt = (x2, x2 + c2)
                        if all(pt != p for p in pts):
                            pts.append(pt)
                            break
            m3 = random.randint(-3, 3)
            if m3 == 0:
                m3 = 1
            b3 = random.randint(-5, 5)
            for x in random.sample(range(-10, 10), 4):
                pt = (x, m3 * x + b3)
                if all(pt != p for p in pts):
                    pts.append(pt)
                else:
                    for x3 in range(-15, 15):
                        pt = (x3, m3 * x3 + b3)
                        if all(pt != p for p in pts):
                            pts.append(pt)
                            break
            if len(pts) != 12 or not _is_valid(pts):
                continue
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            if time.time() < deadline - 10:
                l2, s2 = _local_search(pts, deadline)
                if s2 < best_score:
                    best_score = s2
                    best_points = l2

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
