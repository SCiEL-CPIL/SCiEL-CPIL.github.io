import random
import math

def count_collinear_fast(points):
    """
    Returns (c3, c4) where c3 is number of 3-collinear triples and c4 is number of 4-collinear quadruples.
    Uses O(N^2) slope counting.
    """
    n = len(points)
    c3_total = 0
    c4_total = 0
    
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize slope (dx, dy)
            # Handle vertical lines
            if dx == 0:
                key = (0, 1)
            elif dy == 0:
                key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Ensure consistent sign
                if dx < 0:
                    dx = -dx
                    dy = -dy
                key = (dx, dy)
            if key not in slopes:
                slopes[key] = 0
            slopes[key] += 1
        
        for count in slopes.values():
            # count is number of other points on this line through i
            # Total points on this line = count + 1
            # Number of 3-subsets containing i on this line = C(count, 2)
            c3_total += count * (count - 1) // 2
            # Number of 4-subsets containing i on this line = C(count, 3)
            if count >= 3:
                c4_total += count * (count - 1) * (count - 2) // 6
                
    # Each 3-collinear triple is counted 3 times (once for each point)
    c3_total //= 3
    # Each 4-collinear quadruple is counted 4 times (once for each point)
    c4_total //= 4
    
    return c3_total, c4_total

def construct():
    random.seed(42)
    n = 20
    # Initialize with random points
    points = []
    for _ in range(n):
        points.append((random.randint(-100, 100), random.randint(-100, 100)))
    
    # Ensure distinct
    while len(set(points)) < n:
        points[random.randrange(n)] = (random.randint(-100, 100), random.randint(-100, 100))
        
    c3, c4 = count_collinear_fast(points)
    best_energy = 1000 * c4 - c3
    best_points = points[:]
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.995
    steps = 50000
    
    for step in range(steps):
        if T < 0.01:
            break
            
        i = random.randrange(n)
        old_p = points[i]
        new_p = (random.randint(-100, 100), random.randint(-100, 100))
        
        if new_p in points:
            continue
            
        points[i] = new_p
        
        new_c3, new_c4 = count_collinear_fast(points)
        new_energy = 1000 * new_c4 - new_c3
        
        delta = new_energy - best_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            best_energy = new_energy
            best_points = points[:]
        else:
            points[i] = old_p
            
        T *= cooling
        
    return [[p[0], p[1]] for p in best_points]
