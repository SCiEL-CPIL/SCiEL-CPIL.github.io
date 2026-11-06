import time
import random
import math

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly 2 points from the given list.
    points: list of (x, y) tuples.
    """
    n = len(points)
    if n < 2:
        return 0
        
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line equation: a*x + b*y + c = 0
            # a = y1 - y2
            # b = x2 - x1
            # c = x1*y2 - x2*y1
            
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line coefficients (a, b, c)
            # We want a unique representation for the same line.
            # 1. Ensure gcd(a, b, c) is 1.
            # 2. Ensure the first non-zero coefficient is positive.
            
            if a == 0 and b == 0:
                # Should not happen for distinct points
                continue
                
            # Calculate GCD
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            
            # Normalize sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for count in line_counts.values():
        # A line with k points has C(k, 2) pairs.
        # If count == 1, then C(k, 2) = 1 => k = 2.
        # So this is an ordinary line.
        if count == 1:
            ordinary += 1
            
    return ordinary

def is_all_collinear(points):
    """
    Checks if all points are collinear.
    """
    if len(points) <= 2:
        return True
        
    # Take first 3 distinct points to define a line
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    
    # Check if p0, p1, p2 are collinear
    # Cross product of (p1-p0) and (p2-p0)
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    
    if cross != 0:
        return False
        
    # If the first 3 are collinear, check if all others are on the same line
    # Line through p0 and p1: (y1-y0)x - (x1-x0)y + (x1*y0 - x0*y1) = 0
    a = p1[1] - p0[1]
    b = p0[0] - p1[0]
    c = p1[0]*p0[1] - p0[0]*p1[1]
    
    for p in points:
        if a*p[0] + b*p[1] + c != 0:
            return False
            
    return True

def construct():
    # Set a deadline to ensure we return a valid result within time limits
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin for 150s limit
    
    # Initial configuration: 
    # Try a configuration close to Böröczky or a grid.
    # A 3x4 grid has many ordinary lines.
    # Let's start with points on a parabola y = x^2, which has 0 ordinary lines if n is small?
    # For points on a parabola, any line intersects in at most 2 points.
    # So ALL lines defined by pairs of points are ordinary.
    # Number of lines = C(n, 2). For n=12, C(12,2) = 66.
    # This is bad. We want FEW ordinary lines.
    # We want many points on the same line.
    
    # Good start: 6 points on one line, 6 on another?
    # If we have 6 points on line L1 and 6 on line L2.
    # Lines with 2 points:
    # - Lines connecting a point on L1 to a point on L2. There are 6*6=36 such lines.
    # - Are there any other lines?
    # - If no 3 points are collinear across L1 and L2, then these 36 are ordinary.
    # - Total ordinary = 36.
    
    # Better: Böröczky configuration.
    # For n=12, a known configuration has 6 ordinary lines?
    # Actually, the minimum for n=12 is not 6. 
    # The Böröczky configuration B_n has n/2 ordinary lines for even n?
    # For n=6, B_6 has 3 ordinary lines.
    # For n=8, B_8 has 4 ordinary lines.
    # For n=12, B_12 has 6 ordinary lines.
    # So the target is 6.
    
    # We need to find integer coordinates for a Böröczky-like configuration.
    # B_n consists of n/2 points on a conic (parabola) and n/2 points at infinity (or on a line at infinity in projective plane, which maps to parallel lines in Euclidean).
    # In Euclidean integers, "points at infinity" are not directly representable.
    # However, we can use a large circle or parabola and points "far away".
    
    # Alternative: Use a set of points with high collinearity.
    # Start with a random set of 12 points with small coordinates.
    
    def random_point(bound):
        return (random.randint(-bound, bound), random.randint(-bound, bound))
    
    # Initialize with a diverse set
    points = []
    while len(points) < 12:
        p = random_point(100)
        if p not in points:
            points.append(p)
            
    # Ensure not all collinear
    while is_all_collinear(points):
        # Perturb one point
        idx = random.randint(0, 11)
        points[idx] = random_point(100)
        if points[idx] in points:
             # Handle collision
             points[idx] = (points[idx][0] + 1, points[idx][1])
             if points[idx] in points:
                 points[idx] = (points[idx][0] + 2, points[idx][1])

    best_points = points[:]
    best_score = count_ordinary_lines(best_points)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.99
    
    # Pre-allocate for speed? No, list is fine.
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a move
        i = random.randint(0, 11)
        old_pt = best_points[i]
        
        # Choose a move size. Start large, shrink later.
        # If T is high, allow large jumps.
        move_bound = int(1000 * (T / 100.0)) + 10
        if move_bound > 10**6:
            move_bound = 10**6
            
        dx = random.randint(-move_bound, move_bound)
        dy = random.randint(-move_bound, move_bound)
        
        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
        
        # Check constraints
        if new_pt == old_pt:
            continue
            
        # Check distinctness
        if new_pt in best_points:
            continue
            
        # Create trial state
        trial_points = best_points[:]
        trial_points[i] = new_pt
        
        # Check collinearity
        if is_all_collinear(trial_points):
            continue
            
        # Calculate new score
        new_score = count_ordinary_lines(trial_points)
        
        delta = new_score - best_score
        
        if delta < 0:
            # Improvement
            best_points = trial_points
            best_score = new_score
        else:
            # Accept with probability exp(-delta / T)
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
                
            if random.random() < prob:
                best_points = trial_points
                best_score = new_score
                
        T *= cooling
        
        # If we found 6, we can stop early as it's the theoretical lower bound
        if best_score <= 6:
            break
            
    # Final check and cleanup
    # Ensure coordinates are within 10^9
    # Our moves are bounded, so this should be fine.
    
    return [list(p) for p in best_points]

if __name__ == "__main__":
    # Quick self-test
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", count_ordinary_lines([tuple(p) for p in pts]))
