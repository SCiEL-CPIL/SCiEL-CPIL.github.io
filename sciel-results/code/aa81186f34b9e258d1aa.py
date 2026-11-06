"""Improved program: start from 4x4 grid (score 10) and local-search for more 4-point lines."""
import time
import random

_DEADLINE_MARGIN = 5.0

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _canonical_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, x1)
    if dy == 0:
        return (1, y1)
    g = _gcd(abs(dx), abs(dy))
    dx_r = dx // g
    dy_r = dy // g
    if dx_r < 0:
        dx_r = -dx_r
        dy_r = -dy_r
    c_num = y1 * dx_r - dy_r * x1
    return (1, (dy_r, dx_r, c_num))

def _build_line_map(pts):
    n = len(pts)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line(pts[i], pts[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def _count_lines_exactly_4(pts):
    line_map = _build_line_map(pts)
    count = 0
    for key, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _has_five_collinear(pts):
    line_map = _build_line_map(pts)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return True
    return False

def _check_valid(pts):
    if len(pts) != 16:
        return False
    seen = set()
    for p in pts:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return not _has_five_collinear(pts)

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _random_point(exclude):
    while True:
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        if (x, y) not in exclude:
            return [x, y]

def _local_search(pts, deadline):
    best_pts = [list(p) for p in pts]
    best_score = _count_lines_exactly_4(best_pts)
    cur_pts = [list(p) for p in pts]
    cur_score = best_score
    n = 16
    
    while time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            orig = list(cur_pts[idx])
            exclude = set(tuple(p) for p in cur_pts)
            for _ in range(30):
                if time.time() > deadline:
                    break
                new_pt = _random_point(exclude)
                cur_pts[idx] = new_pt
                if _has_five_collinear(cur_pts):
                    cur_pts[idx] = orig
                    continue
                score = _count_lines_exactly_4(cur_pts)
                if score > cur_score:
                    cur_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_pts = [list(p) for p in cur_pts]
                    break
                cur_pts[idx] = orig
        if not improved:
            # Simulated-annealing-like perturbation
            for _ in range(10):
                if time.time() > deadline:
                    break
                idx = random.randrange(n)
                orig = list(cur_pts[idx])
                exclude = set(tuple(p) for p in cur_pts)
                new_pt = _random_point(exclude)
                cur_pts[idx] = new_pt
                if _has_five_collinear(cur_pts):
                    cur_pts[idx] = orig
                    continue
                score = _count_lines_exactly_4(cur_pts)
                if score >= cur_score:
                    cur_score = score
                    if score > best_score:
                        best_score = score
                        best_pts = [list(p) for p in cur_pts]
                else:
                    cur_pts[idx] = orig
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    pts = _initial_grid()
    best_pts = _local_search(pts, deadline)
    
    if not _check_valid(best_pts):
        best_pts = _initial_grid()
    
    return best_pts
