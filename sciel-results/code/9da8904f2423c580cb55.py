import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Seed points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_integer_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        s = int(math.isqrt(d2))
        return s*s == d2

    def is_collinear(p1, p2, p3):
        # (y2-y1)(x3-x1) == (y3-y1)(x2-x1)
        return (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        # Check if 4 points lie on a circle.
        # Determinant of the matrix:
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # is zero.
        # Or use the property that opposite angles sum to 180, but determinant is safer for exactness.
        def det4(m):
            # m is 4x4
            return (
                m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            )
        
        m = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            m.append([x*x+y*y, x, y, 1])
        return det4(m) == 0

    # Check if the seed set itself is valid (it is, per prompt)
    # Search for a 6th point
    # We expand the search radius gradually
    max_r = 300
    found_6 = None
    
    for r in range(1, max_r + 1):
        if time.time() > deadline:
            break
        for x in range(-r, r + 1):
            if time.time() > deadline:
                break
            for y in range(-r, r + 1):
                p_new = (x, y)
                # Skip if already in seed
                if p_new in seed_points:
                    continue
                
                # Check integer distances to all seed points
                valid_dist = True
                for p_seed in seed_points:
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
                for i in range(len(seed_points)):
                    for j in range(i+1, len(seed_points)):
                        for k in range(j+1, len(seed_points)):
                            if is_concyclic(p_new, seed_points[i], seed_points[j], seed_points[k]):
                                valid_concyclic = False
                                break
                        if not valid_concyclic:
                            break
                    if not valid_concyclic:
                        break
                if not valid_concyclic:
                    continue
                
                # If we reach here, p_new is a valid 6th point
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
