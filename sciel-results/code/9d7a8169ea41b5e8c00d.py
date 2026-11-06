import math
import random
import time

def _count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients a, b, c for ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            # Normalize
            if a == 0 and b == 0:
                continue
                
            # Find GCD
            g = math.gcd(a, math.gcd(b, c))
            if g < 0:
                g = -g
            if g != 0:
                a //= g
                b //= g
                c //= g
                
            # Ensure consistent sign: first non-zero of (a,b,c) is positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
                
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 # Increased to 60s to allow more search within 150s limit
    
    # Initial configurations
    # 1. 4x3 Grid (12 points)
    grid_4x3 = [[i, j] for i in range(4) for j in range(3)]
    
    # 2. 3x4 Grid (12 points)
    grid_3x4 = [[i, j] for i in range(3) for j in range(4)]
    
    # 3. Random (12 distinct points)
    random.seed(42)
    rand_pts = []
    while len(rand_pts) < 12:
        p = [random.randint(-10, 10), random.randint(-10, 10)]
        if p not in rand_pts:
            rand_pts.append(p)
            
    candidates = [grid_4x3, grid_3x4, rand_pts]
    
    best_points = None
    best_score = float('inf')
    
    for init_pts in candidates:
        pts = [p[:] for p in init_pts]
        current_score = _count_ordinary_lines(pts)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in pts]
            
        # Hill Climbing
        no_improve_count = 0
        temp = 10.0
        
        while time.time() < deadline and no_improve_count < 50:
            # Pick random point
            idx = random.randint(0, 11)
            old_x, old_y = pts[idx]
            
            # Propose move: small step
            step = random.randint(-5, 5)
            new_x = old_x + step
            new_y = old_y + random.randint(-5, 5)
            
            # Bounds check (loose, just to keep numbers small)
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
                
            # Distinct check
            if any(pts[i][0] == new_x and pts[i][1] == new_y for i in range(12) if i != idx):
                continue
                
            pts[idx] = [new_x, new_y]
            new_score = _count_ordinary_lines(pts)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in pts]
            else:
                # Revert
                pts[idx] = [old_x, old_y]
                no_improve_count += 1
                # Cool down
                temp *= 0.95
                
            if current_score == 6:
                break # Found theoretical min
                
    return best_points
