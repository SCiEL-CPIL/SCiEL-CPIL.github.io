import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Seed points (known valid 5-point set)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_integer_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        s = int(math.isqrt(d2))
        return s*s == d2

    def is_collinear(p1, p2, p3):
        return (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0])

    # Precompute circle equations for all triplets of seed points
    # Circle: x^2 + y^2 + D*x + E*y + F = 0
    # System for (D,E,F):
    # D*x1 + E*y1 + F = -(x1^2+y1^2)
    # D*x2 + E*y2 + F = -(x2^2+y2^2)
    # D*x3 + E*y3 + F = -(x3^2+y3^2)
    # Solve linear system.
    
    circle_params = []
    for p1, p2, p3 in combinations(seed_points, 3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        b1 = -(x1*x1 + y1*y1)
        b2 = -(x2*x2 + y2*y2)
        b3 = -(x3*x3 + y3*y3)
        
        # Determinant of coeff matrix
        det = x1*(y2*y3 - y3*y2) - y1*(x2*y3 - x3*y2) + (x2*y3 - x3*y2) # Wait, standard Cramer's rule
        # Matrix:
        # x1 y1 1
        # x2 y2 1
        # x3 y3 1
        det = x1*(y2 - y3) - y1*(x2 - x3) + (x2*y3 - x3*y2)
        
        if det == 0:
            # Collinear, no circle (or line). But seed points have no 3 collinear.
            continue
            
        D = (b1*(y2-y3) - y1*(b2-b3) + (b2*y3 - b3*y2)) / det
        E = (x1*(b3-b2) - b1*(x3-x2) + (x3*b2 - x2*b3)) / det
        F = (x1*y2*b3 - x1*y3*b2 - x2*y1*b3 + x2*y3*b1 + x3*y1*b2 - x3*y2*b1) / det
        
        # Store D, E, F as Fractions to be exact? Or just check with integers?
        # If we use integers, we can multiply by det.
        # det * (x^2+y^2) + D_int * x + E_int * y + F_int = 0
        # Let's store the integer coefficients D_int, E_int, F_int
        D_int = b1*(y2-y3) - y1*(b2-b3) + (b2*y3 - b3*y2)
        E_int = x1*(b3-b2) - b1*(x3-x2) + (x3*b2 - x2*b3)
        F_int = x1*y2*b3 - x1*y3*b2 - x2*y1*b3 + x2*y3*b1 + x3*y1*b2 - x3*y2*b1
        
        circle_params.append((D_int, E_int, F_int, det))

    def is_concyclic_fast(p_new, D_int, E_int, F_int, det):
        x, y = p_new
        # Check if det * (x^2+y^2) + D_int*x + E_int*y + F_int == 0
        return det * (x*x + y*y) + D_int * x + E_int * y + F_int == 0

    found_6 = None
    
    # Search for a 6th point F=(x,y)
    # F must be at integer distance from A=(0,0)
    max_dA = 500
    
    for dA in range(1, max_dA + 1):
        if time.time() > deadline:
            break
        dA2 = dA * dA
        for x in range(-dA, dA + 1):
            y2 = dA2 - x*x
            if y2 < 0:
                continue
            y = int(math.isqrt(y2))
            if y*y != y2:
                continue
            
            candidates = [(x, y)]
            if y != 0:
                candidates.append((x, -y))
            
            for p_new in candidates:
                if p_new in seed_points:
                    continue
                
                # Check integer distances to B, C, D, E
                valid_dist = True
                for p_seed in seed_points[1:]:
                    if not is_integer_dist(p_new, p_seed):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                
                # Check no three collinear
                valid_collinear = True
                for i in range(len(seed_points)):
                    for j in range(i+1, len(seed_points)):
                        if is_collinear(p_new, seed_points[i], seed_points[j]):
                            valid_collinear = False
                            break
                    if not valid_collinear:
                        break
                if not valid_collinear:
                    continue
                
                # Check no four concyclic
                valid_concyclic = True
                for D_int, E_int, F_int, det in circle_params:
                    if is_concyclic_fast(p_new, D_int, E_int, F_int, det):
                        valid_concyclic = False
                        break
                if not valid_concyclic:
                    continue
                
                found_6 = p_new
                break
            if found_6:
                break
        if found_6:
            break

    if found_6:
        points = seed_points + [found_6]
    else:
        points = seed_points

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
