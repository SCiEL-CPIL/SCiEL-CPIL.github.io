import time
import random
from collections import defaultdict
from math import gcd as math_gcd

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def count_ordinary_lines_fast(points):
    """Count ordinary lines using line grouping by normalized slope/intercept."""
    n = len(points)
    if n < 2:
        return 0
    
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            a = dy
            b = -dx
            c = -(a * x1 + b * y1)
            
            g = gcd(a, b)
            if g > 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            line_key = (a, b, c)
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    
    ordinary_count = 0
    for line_key, indices in line_points.items():
        if len(indices) == 2:
            ordinary_count += 1
    
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with a known good configuration
    # 3x4 grid has 8 ordinary lines
    initial = [[x, y] for x in range(4) for y in range(3)]
    
    current_pts = [list(p) for p in initial]
    current_score = count_ordinary_lines_fast(current_pts)
    best_pts = [list(p) for p in current_pts]
    best_score = current_score
    
    # Line-aware local search
    # Key insight: move a point to lie on an existing line with 3+ points
    # This can reduce ordinary lines significantly
    
    temp = 3.0
    cooling = 0.998
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_pts[idx]
        
        # Generate candidate positions
        candidates = set()
        
        # 1. Try to place on lines formed by other 11 points that have 3+ points
        line_points = defaultdict(set)
        for i in range(12):
            for j in range(i + 1, 12):
                if i == idx or j == idx:
                    continue
                x1, y1 = current_pts[i]
                x2, y2 = current_pts[j]
                dx = x2 - x1
                dy = y2 - y1
                a = dy
                b = -dx
                c = -(a * x1 + b * y1)
                g = gcd(a, b)
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                line_points[(a, b, c)].add(i)
                line_points[(a, b, c)].add(j)
        
        # For lines with 3+ other points, find integer points on them
        for line_key, other_indices in line_points.items():
            if len(other_indices) >= 3:
                a, b, c = line_key
                # Find integer points on line ax + by = c
                # Try small range of x values
                for x in range(-20, 21):
                    if b == 0:
                        if a != 0 and c % a == 0:
                            x_candidate = c // a
                            y_candidate = 0
                            pt = (x_candidate, y_candidate)
                            if pt != (old_x, old_y) and pt not in [tuple(p) for p in current_pts]:
                                candidates.add(pt)
                    else:
                        val = c - a * x
                        if val % b == 0:
                            y_candidate = val // b
                            pt = (x, y_candidate)
                            if pt != (old_x, old_y) and pt not in [tuple(p) for p in current_pts]:
                                candidates.add(pt)
        
        # 2. Local perturbations
        for _ in range(5):
            nx = old_x + random.randint(-3, 3)
            ny = old_y + random.randint(-3, 3)
            if (nx, ny) != (old_x, old_y) and (nx, ny) not in [tuple(p) for p in current_pts]:
                candidates.add((nx, ny))
        
        # 3. Global random
        for _ in range(3):
            nx = random.randint(-40, 40)
            ny = random.randint(-40, 40)
            if (nx, ny) != (old_x, old_y) and (nx, ny) not in [tuple(p) for p in current_pts]:
                candidates.add((nx, ny))
        
        # Try candidates
        moved = False
        for nx, ny in candidates:
            current_pts[idx] = [nx, ny]
            new_score = count_ordinary_lines_fast(current_pts)
            
            if new_score < current_score:
                current_score = new_score
                moved = True
                if new_score < best_score:
                    best_score = new_score
                    best_pts = [list(p) for p in current_pts]
                break
            else:
                delta = new_score - current_score
                if random.random() < pow(2.718, -delta / temp):
                    current_score = new_score
                    moved = True
                    break
                else:
                    current_pts[idx] = [old_x, old_y]
        
        if not moved:
            current_pts[idx] = [old_x, old_y]
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    return best_pts
