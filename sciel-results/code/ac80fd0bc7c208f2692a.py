import time
import random
import math
from itertools import combinations

def count_exact_four_lines(points):
    """
    Returns the number of lines containing exactly 4 points,
    and a penalty if any line has >= 5 points.
    """
    n = len(points)
    line_map = {}
    
    # Generate all pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By + C = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            # Find GCD of A, B, C
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Ensure canonical form
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in line_map:
                line_map[key] = set([i, j])
            else:
                line_map[key].add(i)
                line_map[key].add(j)
                
    score = 0
    penalty = 0
    for key, point_set in line_map.items():
        k = len(point_set)
        if k == 4:
            score += 1
        elif k >= 5:
            penalty += (k - 4) * 100 # Heavy penalty
    
    return score - penalty

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin for 120s limit
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_exact_four_lines(best_points)
    
    # Local Search with Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 10.0 # Initial temperature
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        i = random.randint(0, 15)
        old_x, old_y = current_points[i]
        
        # Random move: small step or large jump
        if random.random() < 0.5:
            # Small step
            step = random.randint(1, 3)
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
        else:
            # Large jump
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        is_distinct = True
        for j in range(16):
            if j == i: continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Apply move
        current_points[i] = [new_x, new_y]
        new_score = count_exact_four_lines(current_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = [old_x, old_y]
            
        T *= cooling
        if T < 0.1:
            # Reset temperature or restart
            T = 10.0
            # Optional: random restart
            if random.random() < 0.1:
                current_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
                # Ensure distinct
                while len(set(map(tuple, current_points))) < 16:
                    idx = random.randint(0, 15)
                    current_points[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
                current_score = count_exact_four_lines(current_points)
                
    return best_points
