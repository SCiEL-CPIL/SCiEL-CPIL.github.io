import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    n = len(points)
    line_counts = defaultdict(int)
    
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
            line_counts[line_id] += 1
            
    four_count = 0
    five_count = 0
    for line_id, cnt in line_counts.items():
        if cnt >= 5:
            five_count += 1
        elif cnt == 4:
            four_count += 1
            
    # Penalty for any line with >= 5 points
    if five_count > 0:
        return four_count - 100 * five_count
    return four_count

def construct():
    start_time = time.time()
    time_limit = 50  # seconds
    
    # Initialize with random points
    def random_points():
        pts = []
        seen = set()
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        return pts

    best_points = random_points()
    best_score = count_four_lines(best_points)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            # Restart with new random points if stuck
            current_points = random_points()
            current_score = count_four_lines(current_points)
            T = 10.0
        
        # Pick a random point to move
        i = random.randint(0, 15)
        old_x, old_y = current_points[i]
        
        # Generate new position
        # Strategy: 50% random nearby, 50% random far
        if random.random() < 0.5:
            nx = old_x + random.randint(-10, 10)
            ny = old_y + random.randint(-10, 10)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
        
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
            
        # Check for duplicates
        is_dup = False
        for j in range(16):
            if j != i and current_points[j][0] == nx and current_points[j][1] == ny:
                is_dup = True
                break
        
        if is_dup:
            continue
            
        # Apply move
        current_points[i] = [nx, ny]
        new_score = count_four_lines(current_points)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0:
            accept = True
        else:
            # Clamp exponent to avoid underflow/overflow issues
            exp_val = delta / T
            if exp_val < -50:
                accept = False
            else:
                accept = random.random() < math.exp(exp_val)
        
        if accept:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            # Revert
            current_points[i] = [old_x, old_y]
            
        T *= cooling
        
    return best_points
