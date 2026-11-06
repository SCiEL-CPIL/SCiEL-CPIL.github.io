import math
import random
import time

# EVOLVE-BLOCK-START

def _count_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    four_point_lines = 0
    five_point_lines = 0
    seen_lines = set()
    
    # Precompute gcd for speed? No, N is small.
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
                
            # Normalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = math.gcd(dx, dy)
                dx //= g
                dy //= g
                
            # Line equation: A x + B y + C = 0
            # Normal vector (A, B) is perpendicular to (dx, dy)
            A = dy
            B = -dx
            C = -A * x1 - B * y1
            
            # Normalize sign of (A, B, C)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            # Reduce A, B, C by gcd
            g = math.gcd(A, math.gcd(B, C))
            if g > 0:
                A //= g
                B //= g
                C //= g
                
            line_key = (A, B, C)
            
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            # Count points on this line
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if A * xk + B * yk + C == 0:
                    count += 1
                    
            if count == 4:
                four_point_lines += 1
            elif count >= 5:
                five_point_lines += 1
                
    if five_point_lines > 0:
        return -1
        
    return four_point_lines

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Seed configuration
    seed_points = [[x, y] for x in range(4) for y in range(4)]
    seed_score = _count_lines(seed_points)
    
    best_points = seed_points
    best_score = seed_score
    
    # Local Search
    # We will try to improve upon the seed.
    # Strategy: Randomly perturb points in the seed and hill climb.
    
    for attempt in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        # Start from seed and perturb
        current_points = [p[:] for p in seed_points]
        
        # Perturb 1-3 points
        num_perturb = random.randint(1, 3)
        for _ in range(num_perturb):
            idx = random.randint(0, 15)
            current_points[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
            
        # Ensure distinct
        while len(set(map(tuple, current_points))) < 16:
            i = random.randint(0, 15)
            current_points[i] = [random.randint(-10, 10), random.randint(-10, 10)]
            
        current_score = _count_lines(current_points)
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Try moving to nearby locations
            for _ in range(5):
                new_x = random.randint(-10, 10)
                new_y = random.randint(-10, 10)
                
                if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                    continue
                    
                current_points[idx] = [new_x, new_y]
                new_score = _count_lines(current_points)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    break
                else:
                    current_points[idx] = [old_x, old_y]
                    
    return best_points

# EVOLVE-BLOCK-END
