import random
import math
import time
import itertools

def count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    points: list of [x, y]
    """
    n = len(points)
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    count = 0
    for key, pts in line_counts.items():
        if len(pts) == 4:
            count += 1
            
    return count

def is_valid(points):
    """
    Check if points are distinct and no 5 are collinear.
    """
    if len(set(map(tuple, points))) != len(points):
        return False
        
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    for key, pts in line_counts.items():
        if len(pts) >= 5:
            return False
            
    return True

def construct():
    # Initial 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    current_score = count_lines(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    start_time = time.time()
    limit = 110 # seconds
    
    # Simulated Annealing Parameters
    T = 5.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < limit:
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Move generation: Mix of local and global moves
        if random.random() < 0.8:
            # Local move
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        else:
            # Global move within a reasonable box
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
            
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            new_score = count_lines(points)
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
            else:
                # Revert
                points[idx] = [old_x, old_y]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Minimum temperature to allow some exploration
            
    return best_points
