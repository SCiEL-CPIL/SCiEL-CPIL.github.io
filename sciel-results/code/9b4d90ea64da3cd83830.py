import time
import random
import math

def count_exact_four_lines(points):
    n = len(points)
    point_set = set(tuple(p) for p in points)
    
    lines = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = A * x1 + B * y1
            
            if A == 0 and B == 0:
                continue
                
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            g = math.gcd(A, B)
            if g > 0:
                A //= g
                B //= g
                C //= g
                
            lines.add((A, B, C))
            
    four_lines = 0
    for line_key in lines:
        A, B, C = line_key
        count = 0
        for p in points:
            x, y = p
            if A * x + B * y == C:
                count += 1
                if count > 4:
                    return -1
        if count == 4:
            four_lines += 1
            
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    
    best_score = -1
    best_points = None
    
    # 1. Grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    s = count_exact_four_lines(grid)
    if s > best_score:
        best_score = s
        best_points = [p[:] for p in grid]
        
    # 2. Random Initializations
    for _ in range(50):
        pts = []
        while len(pts) < 16:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        s = count_exact_four_lines(pts)
        if s > best_score:
            best_score = s
            best_points = pts
            
    # 3. Local Search on Best
    if best_points is None:
        return grid
        
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try moves
        improved = False
        for _ in range(10):
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            new_point = [new_x, new_y]
            if any(new_point == p for i, p in enumerate(current_points) if i != idx):
                continue
                
            trial_points = [p[:] for p in current_points]
            trial_points[idx] = new_point
            
            trial_score = count_exact_four_lines(trial_points)
            
            if trial_score > current_score:
                current_score = trial_score
                current_points = trial_points
                improved = True
                break
            elif trial_score == current_score and random.random() < 0.05:
                current_points = trial_points
                break
                
        if not improved:
            # Occasionally restart from best known
            if random.random() < 0.1:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                # Try a bigger jump
                idx = random.randint(0, 15)
                current_points[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
                
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
    return best_points
