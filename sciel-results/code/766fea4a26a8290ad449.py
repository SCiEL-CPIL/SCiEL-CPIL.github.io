import random
import time
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line equation: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary_count = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary_count += 1
    return ordinary_count

def _generate_initial():
    # 3x4 grid
    return [[x, y] for x in range(4) for y in range(3)]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = _generate_initial()
    best_score = _count_ordinary(best_points)
    
    # Local search
    # Randomly perturb one point
    max_iter = 5000
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Try a few random new positions
        found_improvement = False
        for _ in range(10):
            # Random jump
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
            # Check bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            # Check distinctness
            new_point = [new_x, new_y]
            is_distinct = True
            for j in range(12):
                if j != idx and current_points[j] == new_point:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            
            # Check not all collinear (simple check: area of triangle with 3 random points != 0)
            # Actually, just check if the new point makes all 12 collinear?
            # If old config was not all collinear, moving one point might make it so.
            # But we want to minimize ordinary lines, so we prefer non-collinear.
            
            # Calculate new score
            current_points[idx] = new_point
            new_score = _count_ordinary(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                found_improvement = True
                break
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
        
        if not found_improvement:
            # Restart from best if stuck for a while?
            # Simple restart
            if random.random() < 0.05:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                
    return best_points
