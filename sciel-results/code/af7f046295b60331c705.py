import time
import math
import random
from collections import defaultdict

_DEADLINE = time.time() + 115.0
_BOUND = 1000


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line constant: x1*dy - y1*dx (normalized by direction)
    c = x1 * dy - y1 * dx
    return (dx, dy, c)


def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            line_map[k].add(i)
            line_map[k].add(j)
    return line_map


def _count_exact4(line_map):
    """Count lines with exactly 4 points."""
    return sum(1 for s in line_map.values() if len(s) == 4)


def _has_five_collinear(line_map):
    """Check if any line has >= 5 points."""
    return any(len(s) >= 5 for s in line_map.values())


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_staggered():
    pts = []
    for row in range(4):
        for col in range(4):
            pts.append([col + row, row])
    return pts


def _seed_affine():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_diagonal():
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


def _seed_hexagonal():
    pts = [[0, 0]]
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
        pts.append([dx, dy])
    for p in [(2,0),(-2,0),(0,2),(0,-2),(2,1),(-2,-1),(1,2),(-1,-2)]:
        pts.append(list(p))
    return pts[:16]


def _seed_3x3_plus():
    pts = [[x, y] for x in range(3) for y in range(3)]
    for e in [(3,1),(-1,1),(1,3),(1,-1),(3,3),(-1,-1),(4,0)]:
        pts.append(list(e))
    return pts


def _local_search_greedy(points, time_budget):
    """Greedy single-point local search with incremental scoring."""
    n = len(points)
    cur = [list(p) for p in points]
    line_map = _build_line_map(cur)
    cur_count = _count_exact4(line_map)
    if _has_five_collinear(line_map):
        return cur, 0  # invalid
    
    best_pts = [list(p) for p in cur]
    best_count = cur_count
    step = 0
    
    while time.time() < time_budget:
        improved = False
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = min(3 + step // 10, 12)
            
            best_local = None
            best_local_count = cur_count - 1  # need strict improvement
            
            # Collect lines through point i
            lines_through_i = set()
            for j in range(n):
                if j != i:
                    lines_through_i.add(_line_key(cur[i], cur[j]))
            
            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue
                    
                    # Incremental update: remove old lines through i, add new
                    # Build new line map by removing point i from all lines, then adding to new lines
                    new_line_map = {}
                    for k, s in line_map.items():
                        if i in s:
                            s2 = s - {i}
                            if s2:
                                new_line_map[k] = s2
                        else:
                            new_line_map[k] = s
                    
                    # Add point i to new lines
                    for j in range(n):
                        if j != i:
                            k = _line_key(np_, cur[j])
                            if k not in new_line_map:
                                new_line_map[k] = set()
                            new_line_map[k].add(i)
                            new_line_map[k].add(j)
                    
                    # Check validity
                    if any(len(s) >= 5 for s in new_line_map.values()):
                        continue
                    
                    cnt = _count_exact4(new_line_map)
                    if cnt > best_local_count:
                        best_local_count = cnt
                        best_local = (np_, new_line_map)
            
            if best_local is not None:
                cur[i] = best_local[0]
                line_map = best_local[1]
                cur_count = best_local_count
                improved = True
                if cur_count > best_count:
                    best_count = cur_count
                    best_pts = [list(p) for p in cur]
        
        step += 1
        if not improved and step > 3:
            # Random kick
            i = random.randrange(n)
            for _ in range(200):
                nx = cur[i][0] + random.randint(-6, 6)
                ny = cur[i][1] + random.randint(-6, 6)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                
                new_line_map = {}
                for k, s in line_map.items():
                    if i in s:
                        s2 = s - {i}
                        if s2:
                            new_line_map[k] = s2
                    else:
                        new_line_map[k] = s
                
                for j in range(n):
                    if j != i:
                        k = _line_key(np_, cur[j])
                        if k not in new_line_map:
                            new_line_map[k] = set()
                        new_line_map[k].add(i)
                        new_line_map[k].add(j)
                
                if any(len(s) >= 5 for s in new_line_map.values()):
                    continue
                
                cnt = _count_exact4(new_line_map)
                if cnt >= cur_count:
                    cur[i] = np_
                    line_map = new_line_map
                    cur_count = cnt
                    if cur_count > best_count:
                        best_count = cur_count
                        best_pts = [list(p) for p in cur]
                    break
    
    return best_pts, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    seeds = [
        _seed_grid,
        _seed_staggered,
        _seed_affine,
        _seed_diagonal,
        _seed_hexagonal,
        _seed_3x3_plus,
    ]
    
    best_overall = None
    best_overall_count = 0
    
    for seed_fn in seeds:
        if time.time() >= deadline:
            break
        
        pts = seed_fn()
        # Scale to use more space
        pts = [[x * 50, y * 50] for x, y in pts]
        
        # Quick validity check
        lm = _build_line_map(pts)
        if _has_five_collinear(lm):
            continue
        
        cnt = _count_exact4(lm)
        
        # Greedy local search with time budget
        time_left = deadline - time.time()
        budget = min(time_left * 0.4, 15.0)
        if budget < 2.0:
            budget = max(budget, 1.0)
        
        result_pts, result_count = _local_search_greedy(pts, time.time() + budget)
        
        if result_count > best_overall_count:
            best_overall_count = result_count
            best_overall = result_pts
    
    # If we still have time, do simulated annealing on the best
    if best_overall is not None and time.time() < deadline - 5:
        sa_budget = deadline - 5 - time.time()
        if sa_budget > 3.0:
            cur = [list(p) for p in best_overall]
            lm = _build_line_map(cur)
            cur_count = _count_exact4(lm)
            if not _has_five_collinear(lm):
                T = 5.0
                min_T = 0.1
                cooling = 0.9995
                stagnant = 0
                best_sa = [list(p) for p in cur]
                best_sa_count = cur_count
                
                while time.time() < deadline - 3:
                    i = random.randrange(16)
                    old = cur[i][:]
                    move_range = 30 if stagnant < 2000 else 80
                    nx = old[0] + random.randint(-move_range, move_range)
                    ny = old[1] + random.randint(-move_range, move_range)
                    nx = max(-_BOUND, min(_BOUND, nx))
                    ny = max(-_BOUND, min(_BOUND, ny))
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue
                    
                    # Incremental
                    new_lm = {}
                    for k, s in lm.items():
                        if i in s:
                            s2 = s - {i}
                            if s2:
                                new_lm[k] = s2
                        else:
                            new_lm[k] = s
                    for j in range(16):
                        if j != i:
                            k = _line_key(np_, cur[j])
                            if k not in new_lm:
                                new_lm[k] = set()
                            new_lm[k].add(i)
                            new_lm[k].add(j)
                    
                    if any(len(s) >= 5 for s in new_lm.values()):
                        continue
                    
                    new_count = _count_exact4(new_lm)
                    delta = new_count - cur_count
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        cur[i] = np_
                        lm = new_lm
                        cur_count = new_count
                        stagnant = 0
                        if cur_count > best_sa_count:
                            best_sa_count = cur_count
                            best_sa = [list(p) for p in cur]
                    else:
                        stagnant += 1
                    
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    
                    if stagnant > 5000 and time.time() < deadline - 10:
                        cur = [list(p) for p in best_sa]
                        lm = _build_line_map(cur)
                        cur_count = best_sa_count
                        stagnant = 0
                        T = 3.0
                
                if best_sa_count > best_overall_count:
                    best_overall = best_sa
                    best_overall_count = best_sa_count
    
    if best_overall is None:
        best_overall = _seed_grid()
    
    return best_overall
