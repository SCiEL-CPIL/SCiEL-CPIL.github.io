import itertools
import random

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Dictionary to store lines as (dx, dy, c) normalized
    # But floating point is bad. Use integer representation.
    # Line through (x1, y1) and (x2, y2) can be represented by (A, B, C)
    # where Ax + By + C = 0.
    # A = y1 - y2
    # B = x2 - x1
    # C = -(A*x1 + B*y1)
    # Normalize (A, B, C) by GCD and sign.
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen
            
            # Find GCD
            g = 1
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure consistent sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    # Start with 3x4 grid
    points = []
    for r in range(3):
        for c in range(4):
            points.append([c, r])
            
    current_count = count_ordinary_lines(points)
    
    # Simple local search: try moving one point to a nearby integer coordinate
    # to see if it reduces the count.
    # This is very slow for 12 points if done naively, but 120s is enough for limited iterations.
    
    best_points = [p[:] for p in points]
    best_count = current_count
    
    # Try a few random perturbations
    for _ in range(1000):
        new_points = [p[:] for p in best_points]
        idx = random.randint(0, 11)
        old_x, old_y = new_points[idx]
        
        # Try a small move
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds and distinctness
        if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
            continue
            
        # Check if point already exists
        exists = False
        for k in range(12):
            if k != idx and new_points[k][0] == new_x and new_points[k][1] == new_y:
                exists = True
                break
        if exists:
            continue
            
        new_points[idx] = [new_x, new_y]
        
        new_count = count_ordinary_lines(new_points)
        if new_count < best_count:
            best_count = new_count
            best_points = new_points
            
    return best_points
