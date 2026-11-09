import math
import random
import time

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    if A == 0 and B == 0:
        return None
        
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def score_points(points):
    line_counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
            
    score = 0
    is_valid = True
    for count in line_counts.values():
        # count is C(k,2)
        # k=4 => 6
        # k=5 => 10
        if count == 6:
            score += 1
        elif count >= 10:
            is_valid = False
            
    return score, is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    grid = [[x, y] for x in range(4) for y in range(4)]
    
    def random_init():
        pts = []
        while len(pts) < 16:
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    initial_configs = [grid, random_init(), random_init()]
    
    best_points = None
    best_score = -1
    
    for config in initial_configs:
        if time.time() > deadline:
            break
            
        current_points = [p[:] for p in config]
        current_score, current_valid = score_points(current_points)
        
        if current_valid and current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            i = random.randint(0, 15)
            old_point = current_points[i]
            
            if random.random() < 0.5:
                dx = random.randint(-50, 50)
                dy = random.randint(-50, 50)
                new_x = max(-1000, min(1000, old_point[0] + dx))
                new_y = max(-1000, min(1000, old_point[1] + dy))
            else:
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
                
            new_point = [new_x, new_y]
            
            if any(new_point == p for j, p in enumerate(current_points) if j != i):
                continue
                
            current_points[i] = new_point
            new_score, new_valid = score_points(current_points)
            
            if new_valid:
                if new_score >= current_score:
                    accept = True
                else:
                    delta = new_score - current_score
                    if T > 0:
                        prob = math.exp(delta / T)
                        accept = random.random() < prob
                    else:
                        accept = False
                
                if accept:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[i] = old_point
            else:
                current_points[i] = old_point
                
            T *= cooling
            
    if best_points is None:
        return grid
        
    return best_points
