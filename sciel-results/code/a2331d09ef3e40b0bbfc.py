"""Minimize ordinary lines for 12 integer points via local search."""
import time
import random
import math
from itertools import combinations

DEADLINE = time.time() + 135  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    # Normalize: make dx > 0, reduce by gcd
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # A line with k points is counted C(k,2) times in the pairs.
    # Ordinary lines are those with exactly 2 points, i.e., counted once.
    # But we need to be careful: line_counts[key] = number of pairs on that line.
    # If a line has m points, it contributes C(m,2) to the count.
    # We want lines where m == 2, i.e., line_counts[key] == 1.
    ordinary = sum(1 for v in line_counts.values() if v == 1)
    return ordinary

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) < 3:
        return True
    p0, p1, p2 = points[0], points[1], points[2]
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    # All three collinear, check all
    for i in range(3, 12):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _clamp(p, limit=10**9):
    x, y = p
    x = max(-limit, min(limit, x))
    y = max(-limit, min(limit, y))
    return [x, y]

def construct():
    random.seed(42)
    
    # Start with a 3x4 grid as initial seed
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    # Try several random starting configurations
    for seed in range(100):
        if time.time() > DEADLINE - 5:
            break
        random.seed(seed + 42)
        points = []
        for _ in range(12):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            points.append([x, y])
        if not _is_valid(points):
            continue
        score = count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
    
    # Local search: try moving each point to nearby integer positions
    points = [p[:] for p in best_points]
    score = best_score
    
    # Also try structured seeds
    structured_seeds = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two lines
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)],
        # Points on three parallel lines
        [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)],
        # Hexagon-like
        [[3, 0], [1, 2], [-1, 2], [-3, 0], [-1, -2], [1, -2],
         [6, 0], [2, 4], [-2, 4], [-6, 0], [-2, -4], [2, -4]],
        # 12 points on a circle (approximate with integers)
        [[7, 24], [15, 8], [8, 15], [-7, 24], [-15, 8], [-8, 15],
         [7, -24], [15, -8], [8, -15], [-7, -24], [-15, -8], [-8, -15]],
        # Böröczky-like: 4 points on a line, 4 on another, 4 on a third
        # Arranged to create many 3-point lines
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 3], [1, 3], [2, 3], [3, 3]],
        # Points forming many diagonal lines
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 2], [1, 2], [2, 2], [3, 2]],
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # Points on 4 lines of 3
        [[0, 0], [1, 0], [2, 0],
         [0, 1], [1, 1], [2, 1],
         [0, 2], [1, 2], [2, 2],
         [0, 3], [1, 3], [2, 3]],
    ]
    
    for s in structured_seeds:
        if time.time() > DEADLINE - 5:
            break
        if not _is_valid(s):
            continue
        sc = count_ordinary_lines(s)
        if sc < best_score:
            best_score = sc
            best_points = [p[:] for p in s]
            points = [p[:] for p in s]
    
    # Local search from best
    points = [p[:] for p in best_points]
    score = count_ordinary_lines(points)
    
    # Simulated annealing / hill climbing with single-point moves and 2-point swaps
    step_size = 10
    for iteration in range(50000):
        if time.time() > DEADLINE - 5:
            break
        if score <= 6:  # theoretical minimum
            break
        
        # Choose move type: 80% single point, 20% two-point swap
        if random.random() < 0.8:
            idx = random.randint(0, 11)
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue
            old_x, old_y = points[idx]
            new_x, new_y = old_x + dx, old_y + dy
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            # Check distinctness
            new_pt = [new_x, new_y]
            valid = True
            for i in range(12):
                if i != idx and points[i] == new_pt:
                    valid = False
                    break
            if not valid:
                continue
            points[idx] = new_pt
            new_score = count_ordinary_lines(points)
            if new_score <= score:
                score = new_score
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
        else:
            # Two-point swap
            idx1, idx2 = random.sample(range(12), 2)
            x1, y1 = points[idx1]
            x2, y2 = points[idx2]
            dx1 = random.randint(-step_size, step_size)
            dy1 = random.randint(-step_size, step_size)
            dx2 = random.randint(-step_size, step_size)
            dy2 = random.randint(-step_size, step_size)
            if (dx1 == 0 and dy1 == 0) and (dx2 == 0 and dy2 == 0):
                continue
            nx1, ny1 = x1 + dx1, y1 + dy1
            nx2, ny2 = x2 + dx2, y2 + dy2
            if abs(nx1) > 10**9 or abs(ny1) > 10**9 or abs(nx2) > 10**9 or abs(ny2) > 10**9:
                continue
            # Check distinctness
            new_pts = [p[:] for p in points]
            new_pts[idx1] = [nx1, ny1]
            new_pts[idx2] = [nx2, ny2]
            valid = len(set(tuple(p) for p in new_pts)) == 12
            if not valid:
                continue
            new_score = count_ordinary_lines(new_pts)
            if new_score <= score:
                points = new_pts
                score = new_score
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
        
        # Occasionally decrease step size
        if iteration % 5000 == 0 and step_size > 1:
            step_size = max(1, step_size // 2)
    
    # Final polish: larger steps to escape local minima, then refine
    if score > 6 and time.time() < DEADLINE - 10:
        points = [p[:] for p in best_points]
        score = best_score
        for step_size in [50, 100, 200, 500, 1000, 5000, 10000]:
            if time.time() > DEADLINE - 5:
                break
            if score <= 6:
                break
            for _ in range(2000):
                if time.time() > DEADLINE - 5:
                    break
                if score <= 6:
                    break
                idx = random.randint(0, 11)
                dx = random.randint(-step_size, step_size)
                dy = random.randint(-step_size, step_size)
                if dx == 0 and dy == 0:
                    continue
                old_x, old_y = points[idx]
                new_x, new_y = old_x + dx, old_y + dy
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                new_pt = [new_x, new_y]
                valid = True
                for i in range(12):
                    if i != idx and points[i] == new_pt:
                        valid = False
                        break
                if not valid:
                    continue
                points[idx] = new_pt
                new_score = count_ordinary_lines(points)
                if new_score <= score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                else:
                    points[idx] = [old_x, old_y]
    
    # Final fine polish with small steps
    if score > 6 and time.time() < DEADLINE - 5:
        points = [p[:] for p in best_points]
        score = best_score
        for _ in range(10000):
            if time.time() > DEADLINE - 3:
                break
            if score <= 6:
                break
            idx = random.randint(0, 11)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            old_x, old_y = points[idx]
            new_x, new_y = old_x + dx, old_y + dy
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            new_pt = [new_x, new_y]
            valid = True
            for i in range(12):
                if i != idx and points[i] == new_pt:
                    valid = False
                    break
            if not valid:
                continue
            points[idx] = new_pt
            new_score = count_ordinary_lines(points)
            if new_score <= score:
                score = new_score
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
    
    return best_points
