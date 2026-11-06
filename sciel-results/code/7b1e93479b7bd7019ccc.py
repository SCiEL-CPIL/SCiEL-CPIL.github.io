import time
import random
from itertools import combinations

def count_ordinary_lines(points):
    """
    Counts the number of lines that contain exactly two of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Dictionary to store lines by a canonical representation
    # A line can be represented by (A, B, C) for Ax + By + C = 0
    # Normalized: gcd(A, B, C) = 1, A > 0 or (A == 0 and B > 0)
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            if A == 0 and B == 0:
                continue # Should not happen for distinct points
            
            g = __import__('math').gcd(A, B)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure canonical sign
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            # If A and B are 0, it's a point, but we handle distinct points.
            # If A=0, B=1 (horizontal)
            # If B=0, A=1 (vertical)
            
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = 0
            line_counts[key] += 1
            
    # Count lines with exactly 2 points (i.e., count == 1 pair)
    # Wait, line_counts[key] is the number of pairs on that line.
    # If a line has k points, it has k*(k-1)/2 pairs.
    # We want lines where the number of points is exactly 2.
    # This corresponds to pair_count == 1.
    
    ordinary_count = 0
    for key, pair_count in line_counts.items():
        if pair_count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial heuristic: 3x4 grid
    grid_points = [[x, y] for x in range(4) for y in range(3)]
    best_points = [p[:] for p in grid_points]
    best_score = count_ordinary_lines(best_points)
    
    # Local Search
    # We will try to improve the score by moving points.
    # We use a simple hill climbing with random restarts.
    
    max_restarts = 100
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Initialize current solution
        if restart_count == 1:
            current_points = [p[:] for p in best_points]
        else:
            # Random initialization
            current_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
            # Ensure distinct
            while len(set(tuple(p) for p in current_points)) < 12:
                idx = random.randint(0, 11)
                current_points[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        
        current_score = count_ordinary_lines(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break # Found theoretical minimum
            
        # Hill Climbing
        no_improve_count = 0
        while time.time() < deadline and no_improve_count < 500:
            idx = random.randint(0, 11)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            
            if dx == 0 and dy == 0:
                continue
                
            new_x = current_points[idx][0] + dx
            new_y = current_points[idx][1] + dy
            
            # Check bounds (though not strictly required if we just minimize, 
            # but keeping them small helps integer stability)
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
                
            # Check distinctness
            new_point = [new_x, new_y]
            is_distinct = True
            for k in range(12):
                if k == idx:
                    continue
                if current_points[k] == new_point:
                    is_distinct = False
                    break
            
            if not is_distinct:
                no_improve_count += 1
                continue
                
            # Calculate new score
            old_point = current_points[idx][:]
            current_points[idx] = new_point
            
            new_score = count_ordinary_lines(current_points)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    if best_score == 6:
                        break
            else:
                # Revert
                current_points[idx] = old_point
                no_improve_count += 1
                
        if best_score == 6:
            break
            
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Score:", count_ordinary_lines(pts))
    print("Points:", pts)
