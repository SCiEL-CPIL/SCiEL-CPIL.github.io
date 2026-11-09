"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
from collections import defaultdict
import math

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_four_lines(points):
    """Count lines with exactly 4 points, return 0 if any line has 5+."""
    n = len(points)
    if n < 4:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                dx_n, dy_n = dx, dy
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            nx = dy_n
            ny = -dx_n
            g2 = _gcd(abs(nx), abs(ny))
            if g2 > 0:
                nx = nx // g2
                ny = ny // g2
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            c = nx * x1 + ny * y1
            
            key = (nx, ny, c)
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for key, pt_set in line_map.items():
        if len(pt_set) >= 5:
            return 0
        if len(pt_set) == 4:
            count += 1
    
    return count

def construct():
    random.seed(42)
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_four_lines(points)
    best_points = [p[:] for p in points]
    
    # Simulated annealing with multiple restarts and better moves
    T = 100.0
    T_min = 0.5
    cooling = 0.998
    max_iters = 80000
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try a random new position - use larger moves sometimes
        if random.random() < 0.3:
            # Large jump
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        else:
            move_range = random.choice([3, 5, 8, 15, 25])
            new_x = old_x + random.randint(-move_range, move_range)
            new_y = old_y + random.randint(-move_range, move_range)
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
        
        # Check if point is already taken
        if any(current_points[i] == [new_x, new_y] for i in range(16) if i != idx):
            continue
        
        current_points[idx] = [new_x, new_y]
        new_score = count_four_lines(current_points)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = [old_x, old_y]
        
        T *= cooling
    
    return best_points

# EVOLVE-BLOCK-END
