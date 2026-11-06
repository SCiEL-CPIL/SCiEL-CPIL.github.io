import time
import random
import math

def count_4_lines(points):
    n = len(points)
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            A = dy
            B = -dx
            C = A * x1 + B * y1
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            
            lines[key].add(i)
            lines[key].add(j)
            
    num_4 = 0
    max_collinear = 0
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt > max_collinear:
            max_collinear = cnt
        if cnt == 4:
            num_4 += 1
            
    return num_4, max_collinear

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    base_points = [[x, y] for x in range(-2, 2) for y in range(-2, 2)]
    current_points = [p[:] for p in base_points]
    current_score, _ = count_4_lines(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 10.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    n_points = 16
    move_radius = 5
    
    stagnation = 0
    max_stagnation = 50000
    
    while time.time() - start_time < time_limit:
        if stagnation > max_stagnation:
            # Reheat and reset to best
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 5.0
            stagnation = 0
            # Optionally try a random restart here if stuck too long
            # But for now, reheat is enough
            
        i = random.randint(0, n_points - 1)
        old_x, old_y = current_points[i]
        
        new_x = old_x + random.randint(-move_radius, move_radius)
        new_y = old_y + random.randint(-move_radius, move_radius)
        
        if new_x < -1000: new_x = -1000
        if new_x > 1000: new_x = 1000
        if new_y < -1000: new_y = -1000
        if new_y > 1000: new_y = 1000
        
        is_distinct = True
        for j in range(n_points):
            if j != i and current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        old_coord = current_points[i]
        current_points[i] = [new_x, new_y]
        
        new_score, max_collinear = count_4_lines(current_points)
        
        if max_collinear >= 5:
            current_points[i] = old_coord
            continue
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                stagnation = 0
            else:
                stagnation += 1
        else:
            current_points[i] = old_coord
            stagnation += 1
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return best_points
