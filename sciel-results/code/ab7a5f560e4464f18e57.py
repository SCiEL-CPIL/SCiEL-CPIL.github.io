import math
import itertools
import random

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Dictionary to store lines as (dx, dy, c) normalized
    # But floating point is bad. Use integer representation.
    # Line through (x1, y1) and (x2, y2) can be represented by (A, B, C)
    # where Ax + By + C = 0.
    # A = y1 - y2
    # B = x2 - x1
    # C = -(A*x1 + B*y1)
    # Normalize (A, B, C) by GCD and sign.
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen
            
            # Find GCD
            g = 1
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure consistent sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    # Simulated Annealing to minimize ordinary lines
    # Initial state: 3x4 grid
    points = []
    for r in range(3):
        for c in range(4):
            points.append([c, r])
    
    current_energy = count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    best_energy = current_energy
    
    # Parameters
    T0 = 100.0  # Initial temperature
    T_end = 0.01
    alpha = 0.99 # Cooling factor
    max_iter = 5000 # Number of iterations
    
    T = T0
    
    for i in range(max_iter):
        # Generate neighbor
        new_points = [p[:] for p in points]
        idx = random.randint(0, 11)
        
        # Move to a random location within bounds
        # We can bias towards smaller coordinates to keep numbers manageable
        # but the problem allows up to 10^9. 
        # Large coordinates don't change line counts, only relative positions matter.
        # So we can restrict search to a smaller box, e.g., -20 to 20.
        new_x = random.randint(-20, 20)
        new_y = random.randint(-20, 20)
        
        # Check distinctness
        if any(k != idx and new_points[k][0] == new_x and new_points[k][1] == new_y for k in range(12)):
            continue
            
        new_points[idx] = [new_x, new_y]
        
        new_energy = count_ordinary_lines(new_points)
        
        # Acceptance criterion
        delta = new_energy - current_energy
        if delta < 0:
            # Always accept improvement
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = [p[:] for p in points]
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                points = new_points
                current_energy = new_energy
        
        T *= alpha
        if T < T_end:
            break
            
    return best_points
