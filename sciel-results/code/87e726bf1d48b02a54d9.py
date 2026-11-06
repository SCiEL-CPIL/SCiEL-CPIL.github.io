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
    # Finite field inspired construction
    # Use GF(4) structure: elements {0, 1, alpha, alpha+1}
    # In GF(4), alpha^2 = alpha + 1
    # Map GF(4) to integers with a nonlinear mapping to preserve field structure
    
    # Try multiple affine transformations of the 4x4 base
    base_configs = []
    
    # Base: 4x4 grid
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    if a * d - b * c == 0:
                        continue  # Singular matrix
                    config = []
                    for x in range(4):
                        for y in range(4):
                            nx = a * x + b * y
                            ny = c * x + d * y
                            config.append([nx, ny])
                    if len(set(tuple(p) for p in config)) == 16:
                        base_configs.append(config)
    
    best_score = -1
    best_points = None
    
    for config in base_configs:
        # Check bounds
        valid = True
        for p in config:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                valid = False
                break
        if not valid:
            continue
            
        score = count_4_lines(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # If we found a good configuration, try to improve with local search
    if best_points is not None and best_score < 15:
        points = [p[:] for p in best_points]
        current_score = best_score
        
        import time
        start_time = time.time()
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        while T > T_min and (time.time() - start_time) < 110:
            i = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            
            new_points = [p[:] for p in points]
            new_points[i][0] += dx
            new_points[i][1] += dy
            
            new_score = count_4_lines(new_points)
            
            if new_score < 0:
                T *= cooling
                continue
                
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    
            T *= cooling
    elif best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = count_4_lines(best_points)
    
    # Final validation
    if count_4_lines(best_points) < 0:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
