import time
import random
import itertools
from math import gcd

DEADLINE = 145  # safety margin before 150s kill

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    """Return (lines, line_count, line_masks) where lines is a list of frozensets of
    collinear point indices (size>=3) and line_masks the bitmask of each line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            s = line_map.get(key)
            if s is None:
                s = set()
                line_map[key] = s
            s.add(i)
            s.add(j)
    lines = []
    line_masks = []
    for s in line_map.values():
        if len(s) >= 3:
            lines.append(frozenset(s))
            m = 0
            for idx in s:
                m |= 1 << idx
            line_masks.append(m)
    return lines, len(lines), line_masks

def _check_no_four(points, lines):
    for L in lines:
        if len(L) >= 4:
            return False
    return True

def _max_cap_free(points, lines, line_masks, line_count, time_limit=5.0):
    """Exact max 3-collinear-free subset via 2-coloring backtracking.

    A subset S is 3-collinear-free iff coloring S=1, rest=0 yields no monochromatic
    collinear triple. We color one point at a time and prune when a line becomes
    monochromatic (all same color) before it is full.
    """
    n = len(points)
    start = time.time()
    timed_out = [False]

    # For each point, the lines it belongs to (by line index)
    point_lines = [[] for _ in range(n)]
    for li, L in enumerate(lines):
        for idx in L:
            point_lines[idx].append(li)

    # Line info
    line_size = [len(L) for L in lines]
    line_mask = list(line_masks)

    state = [0] * line_count   # 0 = undecided, 1 = all-in, -1 = all-out
    color = [0] * n
    best = [0]
    cur = [0]

    def bound(idx):
        # idx = next point to decide (0..n)
        # remaining undecided points
        rem = n - idx
        # lines that can still be decided in-favor (mixed or undecided)
        favor = 0
        for li in range(line_count):
            st = state[li]
            if st == 0:
                favor += 1
            else:
                favor += line_size[li]
        return cur[0] + rem + favor

    def dfs(idx):
        if timed_out[0]:
            return
        if time.time() - start > time_limit:
            timed_out[0] = True
            return
        if idx == n:
            if cur[0] > best[0]:
                best[0] = cur[0]
            return
        if bound(idx) <= best[0]:
            return
        p = idx
        # Try in (color 1)
        ok = True
        touched = []
        for li in point_lines[p]:
            st = state[li]
            if st == -1:
                ok = False
                break
            if st == 0:
                state[li] = 1
                touched.append(li)
        if ok:
            color[p] = 1
            cur[0] += 1
            dfs(idx + 1)
            cur[0] -= 1
            for li in touched:
                state[li] = 0
            color[p] = 0
        if timed_out[0]:
            return
        # Try out (color 0)
        ok = True
        touched = []
        for li in point_lines[p]:
            st = state[li]
            if st == 1:
                ok = False
                break
            if st == 0:
                state[li] = -1
                touched.append(li)
        if ok:
            color[p] = 0
            dfs(idx + 1)
            for li in touched:
                state[li] = 0
        color[p] = 0

    dfs(0)
    return best[0]

def _score(points, time_limit=5.0):
    lines, lc, lm = _build_lines(points)
    if not _check_no_four(points, lines):
        return 999
    return _max_cap_free(points, lines, lm, lc, time_limit)

def _valid(points):
    if len(set(tuple(p) for p in points)) != 20:
        return False
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return False
    lines, lc, lm = _build_lines(points)
    return _check_no_four(points, lines)

def _parabola_init(seed):
    """Generate a good initial configuration: parabola with deterministic jitter."""
    rng = random.Random(seed)
    pts = []
    for i in range(20):
        x = -10 + i
        y = x * x // 4
        # Add small deterministic jitter to break collinearities
        jx = rng.randint(-1, 1)
        jy = rng.randint(-2, 2)
        nx = x + jx
        ny = y + jy
        # Ensure bounds
        nx = max(-100, min(100, nx))
        ny = max(-100, min(100, ny))
        pts.append([nx, ny])
    # Ensure all distinct
    seen = set()
    for i in range(20):
        while tuple(pts[i]) in seen:
            pts[i][0] += 1
            if pts[i][0] > 100:
                pts[i][0] = -100
            if pts[i][1] > 100:
                pts[i][1] = -100
        seen.add(tuple(pts[i]))
    return pts

def _try_move(points, idx, new_pt, lines, lc, lm):
    """Try moving point idx to new_pt. Returns (new_points, new_score) if valid and better, else (None, None)."""
    if abs(new_pt[0]) > 100 or abs(new_pt[1]) > 100:
        return None, None
    # Check collision
    for j in range(20):
        if j != idx and points[j][0] == new_pt[0] and points[j][1] == new_pt[1]:
            return None, None
    # Try the move
    old_pt = points[idx][:]
    points[idx] = new_pt
    new_lines, new_lc, new_lm = _build_lines(points)
    if not _check_no_four(points, new_lines):
        points[idx] = old_pt
        return None, None
    s = _max_cap_free(points, new_lines, new_lm, new_lc, 2.0)
    points[idx] = old_pt
    return [p[:] for p in points], s

def _hill_climb(points, time_budget, rng):
    """Multi-start hill climbing with single-point relocation moves."""
    start = time.time()
    cur = [p[:] for p in points]
    lines, lc, lm = _build_lines(cur)
    if not _check_no_four(cur, lines):
        return cur, 999
    cur_score = _max_cap_free(cur, lines, lm, lc, 3.0)
    best = [p[:] for p in cur]
    best_score = cur_score
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start < time_budget and no_improve < max_no_improve:
        improved = False
        # Try moving each point to a nearby random location
        for _ in range(20):
            if time.time() - start > time_budget:
                break
            idx = rng.randrange(20)
            # Generate candidate move: small perturbation
            for attempt in range(10):
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_pt = [cur[idx][0] + dx, cur[idx][1] + dy]
                new_pts, new_score = _try_move(cur, idx, new_pt, lines, lc, lm)
                if new_score is not None and new_score < best_score:
                    cur = new_pts
                    best_score = new_score
                    best = [p[:] for p in cur]
                    lines, lc, lm = _build_lines(cur)
                    improved = True
                    no_improve = 0
                    break
                elif new_score is not None and new_score == best_score:
                    # Accept neutral moves to escape local optima
                    if rng.random() < 0.1:
                        cur = new_pts
                        lines, lc, lm = _build_lines(cur)
                        break
        if not improved:
            no_improve += 1
            # If stuck, try a larger jump
            if no_improve % 5 == 0:
                idx = rng.randrange(20)
                for attempt in range(20):
                    new_pt = [rng.randint(-100, 100), rng.randint(-100, 100)]
                    new_pts, new_score = _try_move(cur, idx, new_pt, lines, lc, lm)
                    if new_score is not None and new_score < best_score:
                        cur = new_pts
                        best_score = new_score
                        best = [p[:] for p in cur]
                        lines, lc, lm = _build_lines(cur)
                        no_improve = 0
                        break
    return best, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 999
    
    # Multi-start hill climbing from different parabola-based initializations
    num_starts = 15
    time_per_start = (DEADLINE - 20) / num_starts
    
    for s in range(num_starts):
        if time.time() - start_time > DEADLINE - 15:
            break
        pts = _parabola_init(s)
        if not _valid(pts):
            continue
        rng = random.Random(1000 + s)
        refined, rs = _hill_climb(pts, time_per_start, rng)
        if rs < best_score:
            best_score = rs
            best_points = refined
        if best_score <= 15:
            break
    
    # Final verification
    if best_points is None:
        best_points = [[x, x * x // 4] for x in range(-10, 10)]
    
    # Ensure validity
    if not _valid(best_points):
        best_points = [[x, x * x // 4] for x in range(-10, 10)]
    
    return best_points
