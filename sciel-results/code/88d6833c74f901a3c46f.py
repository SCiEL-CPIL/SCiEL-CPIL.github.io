import time
import random
import math

def count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - y2 * x1
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = 0

    ordinary_count = 0
    for key in lines:
        A, B, C = key
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Initial configuration: 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_score = count_ordinary(points)
    best_points = points[:]
    best_score = current_score
    
    # Local search with exhaustive moves in a box
    # Box size: -20 to 20
    box_min = -20
    box_max = 20
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to every position in the box
        for i in range(len(points)):
            orig_p = points[i]
            
            # Generate candidate positions
            # We can skip positions that are the same as orig_p
            # We can also skip positions occupied by other points
            
            for new_x in range(box_min, box_max + 1):
                for new_y in range(box_min, box_max + 1):
                    if time.time() > deadline:
                        return best_points
                    
                    if new_x == orig_p[0] and new_y == orig_p[1]:
                        continue
                    
                    # Check if occupied
                    # Optimization: use a set of points for faster lookup?
                    # For N=12, linear search is fine.
                    is_occupied = False
                    for k, p in enumerate(points):
                        if k != i and p[0] == new_x and p[1] == new_y:
                            is_occupied = True
                            break
                    
                    if is_occupied:
                        continue
                    
                    points[i] = [new_x, new_y]
                    score = count_ordinary(points)
                    
                    if score < current_score:
                        current_score = score
                        if score < best_score:
                            best_score = score
                            best_points = points[:]
                        improved = True
                    else:
                        points[i] = orig_p
            
            # If we found an improvement for this point, we might want to restart the loop
            # or continue. Continuing is fine, but the "current_score" update
            # means subsequent moves are compared against the new score.
            
        if not improved:
            # If no improvement, try a larger box or random jump?
            # Or just break?
            # Let's try a random restart with a different initial config
            # or expand the box slightly.
            
            # Try expanding box
            if box_max < 50:
                box_max += 10
                box_min -= 10
            else:
                # Random restart
                points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
                # Ensure distinct
                pts_set = set()
                while len(pts_set) < 12:
                    p = [random.randint(-10, 10), random.randint(-10, 10)]
                    pts_set.add(tuple(p))
                points = [list(p) for p in pts_set]
                current_score = count_ordinary(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = points[:]
                box_min = -20
                box_max = 20

    return best_points
