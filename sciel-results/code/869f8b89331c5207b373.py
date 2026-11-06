"""Construct 12 integer points minimizing ordinary lines (lines with exactly 2 points)."""

import time
import itertools
import random

# Deadline guard: evaluator kills at 150s, use 120s limit with safety margin
_DEADLINE = 120.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed() - 2.0  # 2s safety margin

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines_fast(points):
    """
    Fast count of ordinary lines.
    Returns the number of lines containing exactly 2 points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
                
            if dx == 0:
                dx, dy = 0, 1
            elif dy == 0:
                dx, dy = 1, 0
            else:
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
            
            # Line key: (dx, dy, c) where dx*y - dy*x = c
            c = dx * y1 - dy * x1
            key = (dx, dy, c)
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
                
    # A line with k points has C(k, 2) pairs.
    # If count == 1, it's an ordinary line.
    # If count > 1, it's not ordinary.
    # We just need to count keys with value 1.
    ordinary = 0
    for count in line_counts.values():
        if count == 1:
            ordinary += 1
            
    return ordinary

def _all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y0) != dy * (x - x0):
            return False
    return True

def _is_valid(points):
    n = len(points)
    if n != 12:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(tuple(p))
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    if _all_collinear(points):
        return False
    return True

def _local_search(points, max_iters=2000):
    """
    Local search with single-point moves and pair swaps.
    """
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines_fast(best)
    
    # Pre-compute small offsets
    offsets = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx != 0 or dy != 0:
                offsets.append((dx, dy))
                
    iters = 0
    while iters < max_iters and _remaining() > 1.0:
        iters += 1
        improved = False
        
        # Try single point moves
        for i in range(12):
            for dx, dy in offsets:
                new_pts = list(best)
                new_pts[i] = [best[i][0] + dx, best[i][1] + dy]
                if not _is_valid(new_pts):
                    continue
                score = _count_ordinary_lines_fast(new_pts)
                if score < best_score:
                    best = new_pts
                    best_score = score
                    improved = True
                    break
            if improved:
                break
                
        if improved:
            continue
            
        # Try pair swaps (move two points)
        # This helps escape local minima
        for _ in range(5):
            i, j = random.sample(range(12), 2)
            dx1 = random.randint(-5, 5)
            dy1 = random.randint(-5, 5)
            dx2 = random.randint(-5, 5)
            dy2 = random.randint(-5, 5)
            new_pts = list(best)
            new_pts[i] = [best[i][0] + dx1, best[i][1] + dy1]
            new_pts[j] = [best[j][0] + dx2, best[j][1] + dy2]
            if not _is_valid(new_pts):
                continue
            score = _count_ordinary_lines_fast(new_pts)
            if score < best_score:
                best = new_pts
                best_score = score
                improved = True
                break
                
        if not improved:
            # Random jump
            i = random.randint(0, 11)
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            new_pts = list(best)
            new_pts[i] = [best[i][0] + dx, best[i][1] + dy]
            if _is_valid(new_pts):
                score = _count_ordinary_lines_fast(new_pts)
                if score < best_score:
                    best = new_pts
                    best_score = score
                else:
                    # Accept with some probability? No, just break if stuck for a while?
                    # For now, break to save time for other restarts
                    break
                    
    return best, best_score

def _generate_initial_points():
    """
    Generate diverse initial points.
    """
    strategy = random.randint(0, 2)
    if strategy == 0:
        # 3x4 Grid
        pts = []
        for x in range(4):
            for y in range(3):
                pts.append([x, y])
        # Perturb
        for i in range(12):
            if random.random() < 0.2:
                pts[i] = [pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2)]
    elif strategy == 1:
        # 4x3 Grid
        pts = []
        for x in range(3):
            for y in range(4):
                pts.append([x, y])
        for i in range(12):
            if random.random() < 0.2:
                pts[i] = [pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2)]
    else:
        # Random cluster
        cx, cy = random.randint(-5, 5), random.randint(-5, 5)
        pts = []
        for i in range(12):
            pts.append([cx + random.randint(-5, 5), cy + random.randint(-5, 5)])
            
    # Ensure distinct
    while len(set(tuple(p) for p in pts)) < 12:
        i = random.randint(0, 11)
        pts[i] = [random.randint(-10, 10), random.randint(-10, 10)]
        
    return pts

def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    random.seed(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if _remaining() < 5.0:
            break
            
        pts = _generate_initial_points()
        result, score = _local_search(pts, max_iters=1000)
        
        if score < best_score:
            best_score = score
            best_points = result
            
        if best_score <= 6:
            break
            
    if best_points is None:
        # Fallback: 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
