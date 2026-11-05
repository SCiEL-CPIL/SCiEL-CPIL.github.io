import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_4lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    
    score = 0
    for key, pairs in lines.items():
        if pairs == 6:
            score += 1
    return score

def _is_valid(points):
    if len(set(map(tuple, points))) != len(points):
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    
    for key, pairs in lines.items():
        if pairs >= 10:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with a subset of 5x5 grid
    # Grid points: -2 to 2
    grid_points = [[x, y] for x in range(-2, 3) for y in range(-2, 3)]
    
    # Pick 16 random points
    current_points = random.sample(grid_points, 16)
    current_score = _count_4lines(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_point = current_points[idx]
        
        # Try to move to a nearby point in the grid, or random
        if random.random() < 0.7:
            # Stay in grid
            candidates = [p for p in grid_points if p != old_point and p not in current_points]
            if candidates:
                new_point = random.choice(candidates)
            else:
                new_point = [random.randint(-2, 2), random.randint(-2, 2)]
        else:
            # Random jump
            new_point = [random.randint(-10, 10), random.randint(-10, 10)]
        
        if new_point in current_points:
            continue
            
        current_points[idx] = new_point
        
        if _is_valid(current_points):
            new_score = _count_4lines(current_points)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            elif new_score == current_score and random.random() < 0.1:
                # Accept equal with prob 0.1
                pass
            else:
                current_points[idx] = old_point
        else:
            current_points[idx] = old_point
            
    return best_points
