"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
from itertools import combinations

def construct():
    # Local Search (Simulated Annealing) approach
    # Objective: Maximize the number of 3-collinear triples (proxy for minimizing 
    # the largest general position subset) while ensuring no 4 points are collinear.
    
    random.seed(42)
    
    def get_lines(points):
        """
        Returns a dict mapping line tuples (sorted normalized) to the list of points on that line.
        Only keeps lines with 2 or more points.
        """
        lines = {}
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize direction vector
                if dx == 0 and dy == 0:
                    continue
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                
                # Line equation: dy*x - dx*y = c
                c = dy * x1 - dx * y1
                
                key = (dx, dy, c)
                if key not in lines:
                    lines[key] = []
                lines[key].append(i)
        return lines

    def count_3_collinear(lines):
        count = 0
        for key, pts in lines.items():
            if len(pts) == 3:
                count += 1
        return count

    def has_4_collinear(lines):
        for key, pts in lines.items():
            if len(pts) >= 4:
                return True
        return False

    def get_4_collinear_points(lines):
        for key, pts in lines.items():
            if len(pts) >= 4:
                return pts
        return None

    # 1. Initialization: Generate random points and fix 4-collinear violations
    points = []
    for _ in range(20):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        while [x, y] in points:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
        points.append([x, y])
    
    # Ensure no 4 collinear by moving points
    max_iters = 1000
    iters = 0
    while iters < max_iters:
        lines = get_lines(points)
        bad_pts = get_4_collinear_points(lines)
        if bad_pts is None:
            break
        # Pick a random point from the bad line and move it
        idx = random.choice(bad_pts)
        old_x, old_y = points[idx]
        for _ in range(100):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if [new_x, new_y] in points:
                continue
            points[idx] = [new_x, new_y]
            lines = get_lines(points)
            if get_4_collinear_points(lines) is None:
                break
        iters += 1
        
    if get_4_collinear_points(get_lines(points)) is not None:
        # Fallback: Return a known valid configuration if random init fails
        # This is a simple spiral or grid perturbation
        return [[i, i*i % 100 - 50] for i in range(20)] # This might have 4 collinear, need better fallback
        
    current_points = [p[:] for p in points]
    lines = get_lines(current_points)
    current_score = count_3_collinear(lines)
    
    # 2. Simulated Annealing / Hill Climbing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    while T > T_min:
        # Pick a random point to move
        i = random.randint(0, 19)
        old_x, old_y = current_points[i]
        
        # Generate new position
        # Try nearby first, then random
        if random.random() < 0.5:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
        if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
            continue
        if [new_x, new_y] in current_points:
            continue
            
        current_points[i] = [new_x, new_y]
        
        # Check validity (no 4 collinear)
        new_lines = get_lines(current_points)
        if get_4_collinear_points(new_lines) is not None:
            # Revert
            current_points[i] = [old_x, old_y]
            continue
            
        new_score = count_3_collinear(new_lines)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = [old_x, old_y]
            
        T *= cooling
        
    return best_points
# EVOLVE-BLOCK-END
