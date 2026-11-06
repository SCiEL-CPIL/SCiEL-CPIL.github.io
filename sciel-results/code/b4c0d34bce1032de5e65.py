import time
import random
import sys
from math import gcd
from itertools import combinations

DEADLINE = 110  # Safety margin before 120s kill

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

def _max_cap_free(points, lines, line_masks, line_count, time_limit=2.0):
    """Exact max 3-collinear-free subset via branch and bound."""
    n = len(points)
    start = time.time()
    timed_out = [False]

    # For each point, the lines it belongs to (by line index)
    point_lines = [[] for _ in range(n)]
    for li, L in enumerate(lines):
        for idx in L:
            point_lines[idx].append(li)

    # Line info: how many points are colored 1 on each line
    line_ones = [0] * line_count
    
    color = [0] * n
    best = [0]
    cur = [0]

    # Precompute: for each line, the sorted list of points on it
    # We process points in an order that helps pruning.
    # Points with many lines are more constrained.
    # Let's sort points by number of lines descending.
    # But we need to maintain the bitmask indices.
    # For simplicity, we stick to 0..n-1 but maybe reorder is better.
    # Let's try reordering points to put highly connected ones first.
    
    # Actually, standard B&B with simple bound is often sufficient for N=20 if lines are sparse.
    # Let's add a tighter bound: 
    # If a line has 2 ones, the 3rd point on that line cannot be added.
    
    def bound(idx):
        # Simple bound: cur + (n - idx)
        return cur[0] + (n - idx)

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
            if line_ones[li] >= 2:
                ok = False
                break
            line_ones[li] += 1
            touched.append(li)
        if ok:
            color[p] = 1
            cur[0] += 1
            dfs(idx + 1)
            cur[0] -= 1
            for li in touched:
                line_ones[li] -= 1
            color[p] = 0
        if timed_out[0]:
            return
        # Try out (color 0)
        color[p] = 0
        dfs(idx + 1)

    dfs(0)
    return best[0]

def _score(points, time_limit=2.0):
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

def _generate_grid_perturbed(seed):
    """Generate points on a 5x4 grid with perturbations to break 4-collinearities."""
    rng = random.Random(seed)
    pts = []
    # 5 columns, 4 rows
    for j in range(4):
        for i in range(5):
            x = i * 10
            y = j * 10
            # Perturb to break collinearity
            # We want to avoid 4 in a row/col/diag
            # Random small offset
            x += rng.randint(-2, 2)
            y += rng.randint(-2, 2)
            pts.append([x, y])
    return pts

def _generate_parabola_perturbed(seed):
    rng = random.Random(seed)
    pts = []
    for i in range(20):
        x = -10 + i
        y = x * x // 4 + rng.randint(-5, 5)
        pts.append([x, y])
    return pts

def _local_search(points, time_budget, rng=None):
    """Hill climbing / Simulated Annealing to minimize score."""
    if rng is None:
        rng = random.Random(42)
    start = time.time()
    cur = [p[:] for p in points]
    lines, lc, lm = _build_lines(cur)
    if not _check_no_four(cur, lines):
        return cur, 999
    cur_score = _max_cap_free(cur, lines, lm, lc, 1.0)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    temperature = 10.0
    cooling = 0.99
    
    while time.time() - start < time_budget:
        if temperature < 0.1:
            break
        temperature *= cooling
        
        # Pick a random point to move
        i = rng.randrange(20)
        ox, oy = cur[i][:]
        
        # Move to a random nearby location
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        nx = ox + dx
        ny = oy + dy
        
        if abs(nx) > 100 or abs(ny) > 100:
            continue
            
        cur[i] = [nx, ny]
        
        # Check validity
        if len(set(tuple(p) for p in cur)) != 20:
            cur[i] = [ox, oy]
            continue
            
        lines, lc, lm = _build_lines(cur)
        if not _check_no_four(cur, lines):
            cur[i] = [ox, oy]
            continue
            
        new_score = _max_cap_free(cur, lines, lm, lc, 1.0)
        
        if new_score < cur_score:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [p[:] for p in cur]
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - cur_score
            if rng.random() < pow(2.71828, -delta / temperature):
                cur_score = new_score
            else:
                cur[i] = [ox, oy]
                
    return best, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 999
    rng = random.Random(12345)
    
    # Try multiple initialization strategies
    generators = [
        (_generate_grid_perturbed, 50),
        (_generate_parabola_perturbed, 20),
    ]

    for gen_func, num_seeds in generators:
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 20:
                break
            pts = gen_func(seed)
            if not _valid(pts):
                continue
            lines, lc, lm = _build_lines(pts)
            s = _max_cap_free(pts, lines, lm, lc, 1.0)
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                
                # If we found a good one, refine it
                if s <= 18 and time.time() - start_time < DEADLINE - 10:
                    remaining = DEADLINE - 10 - (time.time() - start_time)
                    if remaining > 5:
                        refined, rs = _local_search(pts, min(15.0, remaining))
                        if rs < best_score:
                            best_score = rs
                            best_points = refined

    if best_points is None or best_score >= 20:
        # Fallback: simple parabola (score will be 20, but valid)
        best_points = [[x, x * x] for x in range(-10, 10)]
        # Ensure no 4 collinear for fallback? Parabola has no 3 collinear, so no 4.
        
    return best_points
