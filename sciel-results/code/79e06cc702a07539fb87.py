import itertools
import time
import random
from collections import Counter
from math import gcd

def count_exact_four_lines(points):
    """Count lines containing exactly 4 of the 16 points."""
    n = len(points)
    if n < 4:
        return 0
    line_counts = {}
    # For each pair, determine the line (normalized direction + offset)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Line key: direction + a value that identifies the line
            # Use cross product invariant: for line through (x1,y1) with direction (dx,dy)
            # The line is: dy*(x - x1) - dx*(y - y1) = 0
            # So key = (dx, dy, dy*x1 - dx*y1) normalized
            c = dy * x1 - dx * y1
            # Normalize the key so that (dx,dy) is in canonical form
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                c = -c
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Each line with k points is counted C(k,2) times
    # We want lines with exactly 4 points
    four_count = 0
    five_or_more = 0
    for key, pair_count in line_counts.items():
        # pair_count = C(k,2) where k is number of points on line
        # Find k such that C(k,2) = pair_count
        # C(4,2) = 6, C(5,2) = 10
        if pair_count == 6:
            four_count += 1
        elif pair_count >= 10:
            five_or_more += 1
    return four_count, five_or_more

def has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            if dx < 0 or (dx == 0 and dy < 0):
                c = -c
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    for key, pair_count in line_counts.items():
        if pair_count >= 10:  # C(5,2)
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0
    best_five_count = 0
    
    # Start with 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    score, five = count_exact_four_lines(grid)
    best_score = score
    best_points = [p[:] for p in grid]
    best_five_count = five
    
    # Try some structured candidates
    candidates = []
    
    # 4x4 grid shifted/scaled variants
    for sx in [1, 2, 3]:
        for sy in [1, 2, 3]:
            for ox in range(-5, 6):
                for oy in range(-5, 6):
                    pts = [[ox + sx*x, oy + sy*y] for x in range(4) for y in range(4)]
                    if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                        candidates.append(pts)
    
    # Evaluate structured candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        if len(set(tuple(p) for p in pts)) != 16:
            continue
        s, f = count_exact_four_lines(pts)
        if f == 0 and s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            best_five_count = f
    
    # Local search: perturb points
    random.seed(42)
    current = [p[:] for p in best_points]
    current_score = best_score
    
    # Hill climbing with random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            cur = [p[:] for p in best_points]
        else:
            # Random perturbation of best
            cur = [p[:] for p in best_points]
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 15)
                cur[idx][0] += random.randint(-10, 10)
                cur[idx][1] += random.randint(-10, 10)
                if abs(cur[idx][0]) > 1000 or abs(cur[idx][1]) > 1000:
                    cur[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        cur_score, cur_five = count_exact_four_lines(cur)
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving each point
            for idx in range(16):
                if time.time() > deadline:
                    break
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = [p[:] for p in cur]
                        new_pts[idx] = [cur[idx][0] + dx, cur[idx][1] + dy]
                        if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                            continue
                        if len(set(tuple(p) for p in new_pts)) != 16:
                            continue
                        s, f = count_exact_four_lines(new_pts)
                        if f == 0 and s > cur_score:
                            cur = new_pts
                            cur_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if cur_score > best_score:
            best_score = cur_score
            best_points = [p[:] for p in cur]
            best_five_count = cur_five
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    s, f = count_exact_four_lines(final)
    assert f == 0, f"Five collinear found: {f}"
    
    return final
