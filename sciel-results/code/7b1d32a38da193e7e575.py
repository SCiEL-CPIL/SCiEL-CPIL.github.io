import random
import time
import math

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the line direction to handle gcd
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                # Should not happen for distinct points
                continue
            dx_norm = dx // g
            dy_norm = dy // g
            
            # Ensure a consistent sign for the direction
            if dx_norm < 0 or (dx_norm == 0 and dy_norm < 0):
                dx_norm = -dx_norm
                dy_norm = -dy_norm
            
            # Calculate the line key: (normal_vector, c)
            # Line: ax + by = c
            # Normal vector is (-dy, dx) or (dy, -dx)
            # Let's use (dy, -dx) as normal? No, standard is (A, B) for Ax+By=C
            # Direction (dx, dy). Normal (dy, -dx).
            # Equation: dy * x - dx * y = c
            # c = dy * x1 - dx * y1
            
            A = dy_norm
            B = -dx_norm
            C = A * x1 + B * y1
            
            # Normalize (A, B, C)
            g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g2 != 0:
                A //= g2
                B //= g2
                C //= g2
            else:
                # Degenerate case, should not happen
                continue
                
            # Ensure consistent sign for the line key
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
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
    # Initial configuration: 4x3 grid
    # Points: (0,0) to (3,2)
    initial_points = [[x, y] for x in range(4) for y in range(3)]
    best_points = [p[:] for p in initial_points]
    best_count = count_ordinary_lines(best_points)
    
    # Define a search space
    # We will search in a box [-10, 10]
    # But we can also try to "expand" the grid
    
    start_time = time.time()
    deadline = 100.0 # 100 seconds
    
    # Local search
    # We try to move one point at a time
    # Candidate moves:
    # 1. Random position in [-10, 10]
    # 2. Small perturbation of current position
    
    n = 12
    current_points = [p[:] for p in initial_points]
    current_count = best_count
    
    while time.time() < deadline:
        # Pick a random point
        i = random.randint(0, n - 1)
        old_x, old_y = current_points[i]
        
        # Generate a new position
        # Mix of random jumps and local steps
        if random.random() < 0.5:
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
        else:
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
        # Check if new point is distinct from others
        is_distinct = True
        for j in range(n):
            if j == i:
                continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Try the move
        current_points[i] = [new_x, new_y]
        new_count = count_ordinary_lines(current_points)
        
        if new_count < current_count:
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = [old_x, old_y]
            
        # Occasionally, try a larger restart or perturbation
        if random.random() < 0.01:
            # Random restart with a bias towards grid-like structures
            for k in range(n):
                current_points[k] = [random.randint(-5, 5), random.randint(-5, 5)]
            # Ensure distinct
            # Simple check
            for k in range(n):
                for l in range(k+1, n):
                    if current_points[k] == current_points[l]:
                        current_points[l][0] += 1
            current_count = count_ordinary_lines(current_points)
            if current_count < best_count:
                best_count = current_count
                best_points = [p[:] for p in current_points]

    return best_points
