import random
import math
import time

def construct():
    start_time = time.time()
    deadline = 145.0

    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Determinant of the 4x4 matrix for concyclicity
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Using the formula for the determinant
        # A = x1^2+y1^2, B=x1, C=y1, D=1
        # ...
        # For efficiency, we can use the circumcircle equation:
        # x^2 + y^2 + Dx + Ey + F = 0
        # Solve for D, E, F using first 3 points, check if 4th satisfies.
        
        # D, E, F are rational. We can use integer arithmetic by cross-multiplying.
        # Let's stick to the determinant for safety, but optimized.
        # Or simpler: 
        # 2*Area(P1,P2,P3) * ... no, that's for collinear.
        
        # Let's use the property that 4 points are concyclic iff the determinant is 0.
        # We'll compute the determinant explicitly for 4x4.
        # To avoid overflow issues in Python (not an issue with big ints), just compute.
        
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Seed 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point that completes the set
    # We will search in a box around the seed points
    # The seed points are roughly in [-56, 16] x [0, 63]
    # We'll expand the box to [-200, 200] x [-200, 200]
    
    # To make the search faster, we can precompute the required distance squares
    # For a candidate (x,y), we need (x-xi)^2 + (y-yi)^2 to be a square for all i
    
    # Optimization: Iterate x, y.
    # We can also try randomizing the seed points slightly to find other 5-point bases
    # that might be easier to extend.
    
    bases = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (7, 24), (15, 8), (24, 7), (32, 24)], # This one might not be valid, check later
    ]
    
    # Verify bases first
    valid_bases = []
    for base in bases:
        if is_valid_set(base):
            valid_bases.append(base)
            
    if not valid_bases:
        # Fallback to seed if it's valid
        if is_valid_set(seed_points):
            valid_bases.append(seed_points)
        else:
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}

    found_6 = None
    search_box = 300
    
    for base in valid_bases:
        if time.time() > deadline - 10.0:
            break
            
        # Search for 6th point
        # We can restrict the search to integers
        for x in range(-search_box, search_box + 1):
            if time.time() > deadline - 5.0:
                break
            for y in range(-search_box, search_box + 1):
                # Quick check: distance to first point
                dx0 = x - base[0][0]
                dy0 = y - base[0][1]
                d2_0 = dx0*dx0 + dy0*dy0
                if not is_perfect_square(d2_0):
                    continue
                
                # Check distance to other points
                valid_dist = True
                for i in range(1, 5):
                    dx = x - base[i][0]
                    dy = y - base[i][1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break
                
                if valid_dist:
                    candidate = base + [(x, y)]
                    # Full validity check (collinear, concyclic)
                    if is_valid_set(candidate):
                        found_6 = candidate
                        break
            if found_6:
                break
        if found_6:
            break

    if found_6:
        pts = found_6
    else:
        pts = valid_bases[0]

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
