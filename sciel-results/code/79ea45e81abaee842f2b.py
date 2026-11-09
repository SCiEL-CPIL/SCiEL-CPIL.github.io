import math
import time
import random
from math import gcd
from collections import defaultdict

def count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Identify unique lines
    lines = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - y2 * x1
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            lines.add((A, B, C))
            
    # Count points on each line
    ordinary_count = 0
    for A, B, C in lines:
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial configurations
    configs = []
    
    # 1. 3x4 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. Random
    random.seed(42)
    configs.append([ [random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12) ])
    
    best_points = None
    best_score = float('inf')
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    
    # Try each config
    for init_points in configs:
        # Ensure distinct
        seen = set()
        unique_points = []
        for p in init_points:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_points.append(p)
        
        # If not enough points, fill with random
        while len(unique_points) < 12:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_points.append(p)
        
        current_points = [p[:] for p in unique_points]
        current_score = count_ordinary(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        # Annealing
        while time.time() < deadline:
            # Pick a random point to move
            i = random.randint(0, 11)
            orig_p = current_points[i][:]
            
            # Generate new position
            # Small step
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            new_x = orig_p[0] + dx
            new_y = orig_p[1] + dy
            
            # Check bounds and distinctness
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            if [new_x, new_y] in current_points:
                continue
                
            current_points[i] = [new_x, new_y]
            new_score = count_ordinary(current_points)
            
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[i] = orig_p
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                
    return best_points
