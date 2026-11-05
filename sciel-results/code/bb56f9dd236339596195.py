import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    n = len(points)
    line_counts = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope_key = (0, 1)
            elif dy == 0:
                slope_key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                slope_key = (dx_n, dy_n)
            
            c = slope_key[1] * x1 - slope_key[0] * y1
            line_id = (slope_key[0], slope_key[1], c)
            line_counts[line_id].append(i)
    
    four_count = 0
    for line_id, pts in line_counts.items():
        if len(pts) >= 5:
            return -1
        if len(pts) == 4:
            four_count += 1
    
    return four_count

def construct():
    start_time = time.time()
    time_limit = 90  # seconds, leave buffer for overhead
    
    # Start with a random valid configuration or 4x4 grid
    # 4x4 grid is a good local minimum. Let's start there but use SA to escape.
    best = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_four_lines(best)
    
    random.seed(42)
    
    current = [list(p) for p in best]
    current_score = best_score
    
    T = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit and T > 0.01:
        i = random.randint(0, 15)
        orig_x, orig_y = current[i]
        
        # Generate a random new position
        # Mix of small and large jumps
        if random.random() < 0.5:
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
        else:
            nx = orig_x + random.randint(-10, 10)
            ny = orig_y + random.randint(-10, 10)
            
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
            
        # Check collision
        if any(current[j][0] == nx and current[j][1] == ny for j in range(16) if j != i):
            continue
            
        new_pts = [list(p) for p in current]
        new_pts[i] = [nx, ny]
        new_score = count_four_lines(new_pts)
        
        if new_score == -1:
            # Invalid (5 collinear). Reject strongly.
            # In SA, we can treat this as a very high energy cost.
            delta = 100 - new_score # new_score is -1, so delta is 101? 
            # Actually, let's just reject if invalid to maintain constraint feasibility strictly.
            # Or accept with very low prob. Given the constraint is hard, rejecting is safer.
            continue
            
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            delta = current_score - new_score
            if delta < 0:
                # This shouldn't happen if we checked >, but just in case
                pass
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current = new_pts
                    current_score = new_score
        
        T *= cooling
        
    return best
