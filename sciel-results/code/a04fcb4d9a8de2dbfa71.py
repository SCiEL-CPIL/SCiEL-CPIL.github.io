import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    n = len(points)
    line_counts = defaultdict(list)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope_key = (0, 1)
                c = x1
            elif dy == 0:
                slope_key = (1, 0)
                c = y1
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                slope_key = (dx_n, dy_n)
                c = slope_key[1] * x1 - slope_key[0] * y1
                # Wait, the original code used c = slope_key[1]*x1 - slope_key[0]*y1
                # My manual derivation: dy_n * x - dx_n * y = c
                # c = dy_n * x1 - dx_n * y1
                # slope_key is (dx_n, dy_n). So slope_key[0]=dx_n, slope_key[1]=dy_n.
                # c = dy_n * x1 - dx_n * y1.
                # This matches.
            
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
    time_limit = 110
    
    # Start with 4x4 grid
    best = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_four_lines(best)
    
    random.seed(42)
    
    current = [list(p) for p in best]
    current_score = best_score
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = 10.0 # Restart temperature
            
        move_type = random.random()
        
        if move_type < 0.8:
            # Move one point
            i = random.randint(0, 15)
            ox, oy = current[i]
            step = random.randint(1, 15)
            nx = ox + random.randint(-step, step)
            ny = oy + random.randint(-step, step)
            
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            if any(current[j][0] == nx and current[j][1] == ny for j in range(16) if j != i):
                continue
                
            new_pts = [list(p) for p in current]
            new_pts[i] = [nx, ny]
        else:
            # Move two points
            i, j = random.sample(range(16), 2)
            new_pts = [list(p) for p in current]
            
            ox, oy = new_pts[i]
            nx1 = ox + random.randint(-5, 5)
            ny1 = oy + random.randint(-5, 5)
            if abs(nx1) <= 1000 and abs(ny1) <= 1000:
                new_pts[i] = [nx1, ny1]
                
            ox, oy = new_pts[j]
            nx2 = ox + random.randint(-5, 5)
            ny2 = oy + random.randint(-5, 5)
            if abs(nx2) <= 1000 and abs(ny2) <= 1000:
                new_pts[j] = [nx2, ny2]
                
            # Check collisions
            valid = True
            for k in range(16):
                if k in (i, j): continue
                if new_pts[k] == new_pts[i] or new_pts[k] == new_pts[j]:
                    valid = False
                    break
            if new_pts[i] == new_pts[j]:
                valid = False
            if not valid:
                continue
                
        new_score = count_four_lines(new_pts)
        
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            delta = current_score - new_score
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
                
            if random.random() < prob:
                current = new_pts
                current_score = new_score
                
        T *= cooling_rate
        
    return best
