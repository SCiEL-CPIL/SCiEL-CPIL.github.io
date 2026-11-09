import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Normalize line representation
            # Ax + By + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Normalize
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            # Reduce by gcd
            g = __import__('math').gcd(__import__('math').gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary_count = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Random initializations
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
        
    # Local search
    for init_pts in candidates:
        if time.time() > deadline:
            break
            
        current_pts = [list(p) for p in init_pts]
        current_score = count_ordinary_lines(current_pts)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in current_pts]
            
        # Simulated Annealing / Local Search
        temp = 5.0
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = current_pts[idx]
            
            # Move to a nearby integer point
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
            # Ensure distinctness
            new_pt = [new_x, new_y]
            if any(new_pt == p for p in current_pts if p != current_pts[idx]):
                continue
                
            current_pts[idx] = new_pt
            new_score = count_ordinary_lines(current_pts)
            
            # Accept if better or with some probability
            if new_score < current_score or random.random() < pow(2.71828, -(new_score - current_score) / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_pts]
            else:
                current_pts[idx] = [old_x, old_y]
                
            temp *= 0.999
            
    return best_points
