import time
import random
from itertools import combinations

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two points.
    points: list of (x, y)
    """
    n = len(points)
    lines = set()
    ordinary_count = 0
    
    # Map from line equation (normalized) to count of points
    # Line: ax + by + c = 0
    # Normalize (a, b, c) such that gcd(a,b,c)=1 and a>0 or (a==0 and b>0)
    
    line_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            # Find GCD of A, B, C
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a if a > 0 else -a # Handle negative
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Standardize sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    # Now count lines with exactly 2 points
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial configuration: 3x4 grid
    # x in 0..3, y in 0..2
    initial_points = [(x, y) for x in range(4) for y in range(3)]
    current_points = [list(p) for p in initial_points]
    current_score = count_ordinary_lines(current_points)
    
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    # Local search
    # We will try to move points to reduce the score.
    # To speed up, we can limit the search space to a small box around the current grid.
    # Or, we can use a random restart strategy.
    
    # Since the grid is small, we can try all permutations of a larger grid?
    # No, that's too big.
    
    # Let's try moving one point at a time.
    # We'll use a simple hill climbing with random restarts.
    
    def try_move(points, i, new_x, new_y):
        old_x, old_y = points[i]
        if old_x == new_x and old_y == new_y:
            return points, False
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            return points, False
            
        # Check distinctness
        for j in range(len(points)):
            if j != i:
                if points[j][0] == new_x and points[j][1] == new_y:
                    return points, False
        
        points[i][0] = new_x
        points[i][1] = new_y
        return points, True

    # We'll search in a box of size [-10, 10] around the initial grid points.
    # This allows for small perturbations.
    
    while time.time() < deadline:
        # Pick a random point
        i = random.randint(0, 11)
        cx, cy = current_points[i]
        
        # Try random moves in a small neighborhood
        # We'll try a few random steps
        improved = False
        for _ in range(20):
            nx = cx + random.randint(-5, 5)
            ny = cy + random.randint(-5, 5)
            
            new_points, valid = try_move(current_points, i, nx, ny)
            if not valid:
                continue
                
            new_score = count_ordinary_lines(new_points)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                break
            else:
                # Revert
                current_points[i][0] = cx
                current_points[i][1] = cy
        
        if not improved and random.random() < 0.05:
            # Random restart: perturb the whole configuration slightly
            # Or just restart from initial grid
            current_points = [list(p) for p in initial_points]
            current_score = count_ordinary_lines(current_points)
            
    # Return the best points found
    # Ensure they are lists of ints
    return [[int(p[0]), int(p[1])] for p in best_points]

# To be safe, if the local search fails to improve, return the initial grid.
# The initial grid has 15 ordinary lines? Let's assume it's a valid baseline.
