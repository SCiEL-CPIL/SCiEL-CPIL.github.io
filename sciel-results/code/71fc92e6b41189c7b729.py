import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Dictionary to store lines
    # Key: (slope, intercept) or special handling for vertical
    # For integer points, we can use (dy, dx, c) where dy*x - dx*y = c
    # Or simply use a set of frozensets of indices for each line?
    # Simpler: iterate pairs, determine line, count points on it.
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            if dx == 0 and dy == 0:
                continue # Should not happen for distinct points
            
            # Find GCD to normalize
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(dx, dy)
            dx_norm = dx // g
            dy_norm = dy // g
            
            # Equation: dy_norm * x - dx_norm * y = c
            c = dy_norm * x1 - dx_norm * y1
            
            key = (dx_norm, dy_norm, c)
            
            if key not in lines:
                lines[key] = {i, j}
            else:
                lines[key].add(i)
                lines[key].add(j)
    
    ordinary_count = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial configuration: 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_score = count_ordinary_lines(points)
    best_points = points.copy()
    best_score = current_score
    
    # Local search
    # We will try to move points to reduce the count
    # Use a set of candidate moves
    # Since coordinates can be large, we stick to small integers for now
    # or use random jumps
    
    max_iter = 5000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Pick a random point to move
        i = random.randint(0, 11)
        old_x, old_y = points[i]
        
        # Generate a new position
        # Try small perturbations first
        moved = False
        for _ in range(10):
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
            # Ensure distinctness and bounds
            if new_x == old_x and new_y == old_y:
                continue
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            # Check if new point is distinct from others
            duplicate = False
            for j in range(12):
                if j != i:
                    if points[j][0] == new_x and points[j][1] == new_y:
                        duplicate = True
                        break
            if duplicate:
                continue
            
            points[i] = [new_x, new_y]
            new_score = count_ordinary_lines(points)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()
                moved = True
                break
            else:
                # Revert
                points[i] = [old_x, old_y]
        
        if not moved:
            # Try a larger jump or restart
            # Occasionally reset to grid to escape local minima
            if random.random() < 0.1:
                points = [[x, y] for x in range(4) for y in range(3)]
                current_score = count_ordinary_lines(points)
                
    return best_points
