import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s
    
    # Base 5-point set from seed
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_validity(points):
        n = len(points)
        # Check distinct
        pts_set = set(points)
        if len(pts_set) != n:
            return False
            
        # Check distances rational (integers here)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    return False
                    
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Area of triangle
                    area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                    if area2 == 0:
                        return False
                        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        # Circle equation: x^2+y^2+Dx+Ey+F=0
                        # Subtract equations to solve for D, E
                        # (x2^2-x1^2) + (y2^2-y1^2) + D(x2-x1) + E(y2-y1) = 0
                        # (x3^2-x1^2) + (y3^2-y1^2) + D(x3-x1) + E(y3-y1) = 0
                        
                        a1 = x2 - x1
                        b1 = y2 - y1
                        c1 = -(x2*x2 - x1*x1 + y2*y2 - y1*y1)
                        
                        a2 = x3 - x1
                        b2 = y3 - y1
                        c2 = -(x3*x3 - x1*x1 + y3*y3 - y1*y1)
                        
                        det = a1*b2 - a2*b1
                        if det == 0:
                            continue # Collinear, already checked, but skip circle check for this triplet if degenerate
                            
                        D = (c1*b2 - c2*b1) / det
                        E = (a1*c2 - a2*c1) / det
                        
                        # Check if point 4 is on this circle
                        # x4^2 + y4^2 + D*x4 + E*y4 + F = 0
                        # F = -(x1^2 + y1^2 + D*x1 + E*y1)
                        F = -(x1*x1 + y1*y1 + D*x1 + E*y1)
                        
                        val = x4*x4 + y4*y4 + D*x4 + E*y4 + F
                        # Check if val is 0
                        # Using exact rational arithmetic would be better, but float with tolerance might work for small ints
                        # Or use fractions
                        from fractions import Fraction
                        Df = Fraction(c1*b2 - c2*b1, det)
                        Ef = Fraction(a1*c2 - a2*c1, det)
                        Ff = Fraction(-(x1*x1 + y1*y1 + Df*x1 + Ef*y1), 1)
                        
                        valf = Fraction(x4*x4 + y4*y4 + Df*x4 + Ef*y4 + Ff, 1)
                        if valf == 0:
                            return False
                            
        return True

    # Verify base set
    if not check_validity(base_points):
        # Fallback to a known valid 3-point set if base fails (should not happen)
        fallback = [(0,0), (3,0), (0,4)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback]}
    
    best_points = base_points
    
    # Search for 6th point
    # Range search
    # We need integer distances to all 5 base points.
    # This is a strong constraint.
    # Let's try a small range first.
    found_sixth = False
    limit = 200
    for x in range(-limit, limit+1):
        if time.time() > deadline:
            break
        for y in range(-limit, limit+1):
            if time.time() > deadline:
                break
            p_new = (x, y)
            if p_new in base_points:
                continue
                
            # Check distances to base points
            valid_dist = True
            for bx, by in base_points:
                dx = x - bx
                dy = y - by
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    valid_dist = False
                    break
            if not valid_dist:
                continue
                
            # Check collinearity and concyclicality with full set
            candidate_set = base_points + [p_new]
            if check_validity(candidate_set):
                best_points = candidate_set
                found_sixth = True
                break
        if found_sixth:
            break
            
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
