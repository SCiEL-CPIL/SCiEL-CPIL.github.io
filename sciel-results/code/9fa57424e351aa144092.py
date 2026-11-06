import random
import math
from collections import defaultdict

def count_4_lines(points):
    """
    Returns the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check distinctness
    if len(set(map(tuple, points))) != 16:
        return -1
        
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1

    lines = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the slope
            if dx == 0:
                slope = (0, 1) # Vertical
            elif dy == 0:
                slope = (1, 0) # Horizontal
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                # Ensure consistent sign, e.g., dx_n > 0
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                slope = (dx_n, dy_n)
            
            # Calculate intercept
            # Line: A*x + B*y = C
            # A = dy, B = -dx
            # C = A*x1 + B*y1
            # Normalize (A, B, C)
            A = dy
            B = -dx
            C = A*x1 + B*y1
            
            # Normalize the line equation to avoid duplicates
            # Find gcd of A, B, C
            if A == 0 and B == 0:
                continue # Should not happen for distinct points
            
            g_line = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_line == 0:
                g_line = 1
            A_n = A // g_line
            B_n = B // g_line
            C_n = C // g_line
            
            # Ensure canonical form: A_n > 0 or (A_n == 0 and B_n > 0)
            if A_n < 0 or (A_n == 0 and B_n < 0):
                A_n = -A_n
                B_n = -B_n
                C_n = -C_n
                
            key = (A_n, B_n, C_n)
            lines[key].append(i)
            
    count_4 = 0
    for key, pts in lines.items():
        if len(pts) > 4:
            return -1
        if len(pts) == 4:
            count_4 += 1
            
    return count_4

def construct():
    # Initial 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Center it to 0
    for p in points:
        p[0] -= 1.5
        p[1] -= 1.5
    # Scale to integers? No, keep integers.
    # Let's just use 0..3
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_4_lines(points)
    best_points = [p[:] for p in points]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    # Use a fixed seed for reproducibility if needed, but random is fine
    # The task doesn't require deterministic output, just a good one.
    
    import time
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < 110:
        # Pick a random point
        i = random.randint(0, 15)
        # Pick a random new coordinate
        # Stay within bounds
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        # Small perturbation strategy might be better
        # Let's try moving by small steps
        delta_x = random.randint(-2, 2)
        delta_y = random.randint(-2, 2)
        
        new_points = [p[:] for p in points]
        new_points[i][0] += delta_x
        new_points[i][1] += delta_y
        
        new_score = count_4_lines(new_points)
        
        if new_score < 0:
            # Invalid, reject
            continue
            
        delta = new_score - best_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                
        T *= cooling
        
    # Final check
    if count_4_lines(best_points) < 0:
        # Fallback to grid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
