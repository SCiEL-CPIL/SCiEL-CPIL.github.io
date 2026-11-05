import random
from math import gcd
from collections import defaultdict

def count_ordinary(points):
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = dy
            b = -dx
            c = a*x1 + b*y1
            g2 = gcd(abs(a), gcd(abs(b), abs(c)))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            line_counts[(a, b, c)] += 1
    return sum(1 for v in line_counts.values() if v == 2)

def construct():
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    best_points = None
    best_score = float('inf')
    
    def simulated_annealing(initial_points, steps=2000):
        points = [p[:] for p in initial_points]
        current_score = count_ordinary(points)
        best_points_local = [p[:] for p in points]
        best_score_local = current_score
        
        T = 50.0
        cooling = 0.995
        
        for _ in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            i = random.randint(0, 11)
            old_x, old_y = points[i]
            
            # Generate a move
            # Mix of small and medium moves
            if random.random() < 0.8:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            else:
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                
            if dx == 0 and dy == 0:
                continue
                
            new_x = old_x + dx
            new_y = old_y + dy
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            # Check distinctness
            is_distinct = True
            for k in range(12):
                if k == i: continue
                if points[k][0] == new_x and points[k][1] == new_y:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            points[i] = [new_x, new_y]
            new_score = count_ordinary(points)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_points_local = [p[:] for p in points]
            else:
                # Accept with probability
                if random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
                    current_score = new_score
                else:
                    points[i] = [old_x, old_y]
                    
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_points_local, best_score_local

    # Initial configurations
    # 1. Random
    # 2. Grids
    # 3. Lines
    
    configs = []
    
    # Grids
    for r in range(2, 5):
        for c in range(2, 5):
            if r*c == 12:
                pts = [[c_idx, r_idx] for r_idx in range(r) for c_idx in range(c)]
                configs.append(pts)
                
    # Random
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        configs.append(pts)
        
    # Try to put points on 3 lines of 4
    # Line 1: y=0, x=0..3
    # Line 2: y=1, x=1..4
    # Line 3: y=2, x=2..5
    # This creates a parallelogram-like structure
    pts = []
    for i in range(4):
        pts.append([i, 0])
    for i in range(4):
        pts.append([i+1, 1])
    for i in range(4):
        pts.append([i+2, 2])
    configs.append(pts)
    
    for config in configs:
        if len(set(tuple(p) for p in config)) < 12:
            continue
            
        # Run SA multiple times from this config
        for _ in range(5):
            pts, score = simulated_annealing(config, steps=5000)
            if score < best_score:
                best_score = score
                best_points = pts
                
            if time.time() - start_time > time_limit:
                break
        if time.time() - start_time > time_limit:
            break
            
    # If we didn't find anything (shouldn't happen), return a default
    if best_points is None:
        best_points = [[i, 0] for i in range(12)]
        
    return best_points
