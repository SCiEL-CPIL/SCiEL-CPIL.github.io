import random
import time
import math
from itertools import combinations
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    ordinary = sum(1 for pts in line_points.values() if len(pts) == 2)
    return ordinary

def _gcd3(a, b, c):
    g = gcd(abs(a), abs(b))
    return gcd(g, abs(c))

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True

def _affine_plane_3_grid():
    """3x3 grid representing AG(2,3) affine plane."""
    pts = []
    for i in range(3):
        for j in range(3):
            pts.append([j, i])
    return pts

def _construct_12_affine():
    """Deterministic construction: 3x3 grid + 3 points to minimize ordinary lines."""
    base = _affine_plane_3_grid()
    candidates = [
        [[3, 0], [0, 3], [3, 3]],
        [[3, 0], [0, 3], [1, 4]],
        [[3, 0], [0, 3], [2, 4]],
        [[3, 1], [0, 3], [3, 3]],
        [[3, 2], [0, 3], [3, 3]],
        [[-1, 0], [0, 3], [3, 3]],
        [[3, 0], [-1, 3], [3, 3]],
        [[3, 0], [0, 3], [-1, 3]],
    ]
    best_pts = None
    best_score = float('inf')
    for extra in candidates:
        pts = base + extra
        seen = set(tuple(p) for p in pts)
        if len(seen) != 12 or _all_collinear(pts):
            continue
        score = _count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    return best_pts, best_score

def _fast_local_search(points, time_limit):
    """Deterministic hill-climbing with small moves."""
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    n = len(current)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            orig_x, orig_y = current[i]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if any(current[j][0] == new_x and current[j][1] == new_y for j in range(n) if j != i):
                        continue
                    current[i] = [new_x, new_y]
                    new_score = _count_ordinary_lines(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                    else:
                        current[i] = [orig_x, orig_y]
        if not improved or best_score <= 6:
            break
    return best, best_score

def construct():
    deadline = time.time() + 140
    best_points = None
    best_score = float('inf')
    random.seed(42)
    
    # Deterministic affine plane construction
    pts, score = _construct_12_affine()
    if pts:
        best_points = pts
        best_score = score
    
    # Additional initial configs
    configs = [
        _affine_plane_3_grid() + [[3, 0], [0, 3], [3, 3]],
        [[i, 0] for i in range(11)] + [[0, 100]],
        [[j, i] for i in range(3) for j in range(4)],
    ]
    
    for config in configs:
        if time.time() > deadline:
            break
        seen = set(tuple(p) for p in config)
        if len(seen) != 12 or _all_collinear(config):
            continue
        time_left = deadline - time.time()
        if time_left < 5:
            continue
        pts, score = _fast_local_search(config, time_left * 0.9)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            break
    
    if best_points is None:
        return _affine_plane_3_grid() + [[3, 0], [0, 3], [3, 3]]
    return best_points
