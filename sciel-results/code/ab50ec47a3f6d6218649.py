import numpy as np
from math import isqrt
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Seed 5 points
    seed_points = [
        (0, 0),
        (-56, 33),
        (-16, 63),
        (0, 33),
        (16, 63)
    ]
    
    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r*r == n

    def check_geometry(points):
        # points is a list of (x, y)
        n = len(points)
        # Check collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0:
                        return False
        # Check concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        # Determinant of circle equation
                        # | x y x^2+y^2 1 |
                        # | ... | = 0
                        # We can use the property that 4 points are concyclic iff
                        # the determinant of the matrix formed by (x, y, x^2+y^2, 1) is 0.
                        # To avoid large numbers, we can use exact arithmetic or check
                        # if the circle through 1,2,3 passes through 4.
                        # Circle through 1,2,3:
                        # (x1-x2)^2 + (y1-y2)^2 ...
                        # Let's use the determinant method with integers.
                        # D = det(M)
                        # If D == 0, concyclic.
                        
                        # Matrix rows: [x, y, x^2+y^2, 1]
                        # We can compute the determinant of a 4x4 matrix.
                        # Or simpler:
                        # Power of point 4 wrt circle through 1,2,3.
                        # Circle eq: x^2 + y^2 + Dx + Ey + F = 0
                        # Solve for D, E, F using 1,2,3.
                        # Then check if 4 satisfies it.
                        
                        # System:
                        # x1^2+y1^2 + D x1 + E y1 + F = 0
                        # x2^2+y2^2 + D x2 + E y2 + F = 0
                        # x3^2+y3^2 + D x3 + E y3 + F = 0
                        
                        # Subtract 1 from 2 and 3 to eliminate F:
                        # D(x2-x1) + E(y2-y1) = (x1^2+y1^2) - (x2^2+y2^2)
                        # D(x3-x1) + E(y3-y1) = (x1^2+y1^2) - (x3^2+y3^2)
                        
                        # Let A = x2-x1, B = y2-y1, C = x3-x1, Dd = y3-y1
                        # Let R1 = x1^2+y1^2 - x2^2-y2^2
                        # Let R2 = x1^2+y1^2 - x3^2-y3^2
                        
                        # D*A + E*B = R1
                        # D*C + E*Dd = R2
                        
                        # Det = A*Dd - B*C
                        if A*Dd - B*C == 0:
                            continue # Collinear, already checked, but skip circle check if degenerate
                            
                        # D = (R1*Dd - B*R2) / Det
                        # E = (A*R2 - R1*C) / Det
                        # F = -x1^2-y1^2 - D*x1 - E*y1
                        
                        # Check point 4:
                        # x4^2+y4^2 + D*x4 + E*y4 + F == 0
                        # Multiply by Det^2 to clear denominators?
                        # Or just check if:
                        # (x4^2+y4^2)*Det^2 + (D_num*x4 + E_num*y4 + F_num)*Det == 0 ?
                        
                        # Let's use fractions or just integer arithmetic carefully.
                        # D_num = R1*Dd - B*R2
                        # E_num = A*R2 - R1*C
                        # F_num = -(x1^2+y1^2)*Det - D_num*x1 - E_num*y1
                        
                        # Check: (x4^2+y4^2)*Det + D_num*x4 + E_num*y4 + F_num == 0
                        
                        val = (x4*x4 + y4*y4)*Det + D_num*x4 + E_num*y4 + F_num
                        if val == 0:
                            return False
        return True

    def try_find_6th(points_5, limit):
        # Use NumPy to search for a 6th point
        # Grid size 2*limit + 1
        xs = np.arange(-limit, limit+1)
        ys = np.arange(-limit, limit+1)
        X, Y = np.meshgrid(xs, ys)
        
        # Distances squared to each of the 5 points
        # P1 (0,0)
        d1_sq = X*X + Y*Y
        # Filter d1_sq to be perfect squares
        # This is hard to vectorize "is_square" easily without loops or large arrays.
        # But we can filter by checking if sqrt is integer.
        
        # To be efficient, we iterate over possible distances to P1.
        # k1^2 = x^2+y^2 <= (2*limit)^2 * 2
        max_d_sq = (2*limit)**2 * 2
        # Iterate k1
        for k1 in range(1, isqrt(max_d_sq)+1):
            k1_sq = k1*k1
            # Find (x,y) such that x^2+y^2 == k1_sq
            # This is still a loop over x.
            # Maybe just loop over x, y in Python is faster for small limits?
            pass
            
        # Let's just do a Python loop for the search, it's simpler and 160k is fast.
        for x in range(-limit, limit+1):
            for y in range(-limit, limit+1):
                if time.time() > deadline:
                    return None
                
                # Check dist to P1 (0,0)
                d1_sq = x*x + y*y
                if not is_square(d1_sq): continue
                
                # Check dist to P2 (-56,33)
                dx = x - (-56)
                dy = y - 33
                d2_sq = dx*dx + dy*dy
                if not is_square(d2_sq): continue
                
                # Check dist to P3 (-16,63)
                dx = x - (-16)
                dy = y - 63
                d3_sq = dx*dx + dy*dy
                if not is_square(d3_sq): continue
                
                # Check dist to P4 (0,33)
                dx = x - 0
                dy = y - 33
                d4_sq = dx*dx + dy*dy
                if not is_square(d4_sq): continue
                
                # Check dist to P5 (16,63)
                dx = x - 16
                dy = y - 63
                d5_sq = dx*dx + dy*dy
                if not is_square(d5_sq): continue
                
                # All distances are integers.
                # Check geometry
                new_points = points_5 + [(x, y)]
                if check_geometry(new_points):
                    return (x, y)
        return None

    # Try to extend seed with limit 200
    new_point = try_find_6th(seed_points, 200)
    if new_point:
        final_points = seed_points + [new_point]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in final_points]}
        
    # If not found, try limit 500 (might be slow, but we have time)
    if time.time() < deadline - 20:
        new_point = try_find_6th(seed_points, 500)
        if new_point:
            final_points = seed_points + [new_point]
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in final_points]}
            
    # Fallback: Return seed 5 points
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}
