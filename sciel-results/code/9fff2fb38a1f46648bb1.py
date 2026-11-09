import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s buffer
    
    # Seed points
    P1 = (0, 0)
    P2 = (-56, 33)
    P3 = (-16, 63)
    P4 = (0, 33)
    P5 = (16, 63)
    
    base_points = [P1, P2, P3, P4, P5]
    
    # Helper functions
    def is_square(n):
        if n < 0: return False
        r = math.isqrt(n)
        return r*r == n
        
    def dist_sq(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        
    def is_int_dist(p1, p2):
        d2 = dist_sq(p1, p2)
        return is_square(d2)
        
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0
        
    def are_concyclic(p1, p2, p3, p4):
        # Check if 4 points are concyclic
        # Det of matrix:
        # x^2+y^2  x  y  1
        # ...
        # If det is 0, they are concyclic or collinear.
        # We assume collinearity is checked separately.
        import numpy as np
        pts = [p1, p2, p3, p4]
        M = np.array([
            [p[0]**2 + p[1]**2, p[0], p[1], 1] for p in pts
        ])
        # Use rational arithmetic? Numpy float might be imprecise for large ints.
        # Max coord ~100. x^2+y^2 ~ 20000.
        # Det is integer.
        # Let's compute determinant exactly.
        # Since we are in Python, we can use fractions or just integers if we scale?
        # The matrix entries are integers.
        # We can compute determinant using Bareiss algorithm or just sympy?
        # Sympy is not guaranteed, but numpy is.
        # However, for small integers, float determinant might be okay if we check close to 0?
        # Better: exact integer determinant.
        
        # Let's implement a small exact determinant for 4x4.
        def det4(m):
            # m is 4x4 list of lists
            return (
                m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            )

        M = [[p[0]**2 + p[1]**2, p[0], p[1], 1] for p in pts]
        return det4(M) == 0

    def check_validity(points):
        n = len(points)
        # Check all pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                if not is_int_dist(points[i], points[j]):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Search for 6th point
    # We use P1(0,0) and P4(0,33) to constrain y.
    # x^2 + y^2 = k1^2
    # x^2 + (y-33)^2 = k4^2
    # Subtracting: 66y - 1089 = k1^2 - k4^2
    # y = (k1^2 - k4^2 + 1089) / 66
    
    found_point = None
    max_k = 300 # Search range for distances
    
    for k1 in range(1, max_k):
        for k4 in range(1, max_k):
            if time.time() > deadline:
                break
                
            num_y = k1*k1 - k4*k4 + 1089
            if num_y % 66 != 0:
                continue
            
            y = num_y // 66
            
            # x^2 = k1^2 - y^2
            x2 = k1*k1 - y*y
            if x2 < 0:
                continue
            
            if not is_square(x2):
                continue
            
            x = math.isqrt(x2)
            
            # We have two candidates: (x, y) and (-x, y)
            for cx in [x, -x]:
                P6 = (cx, y)
                
                # Check distinctness
                if P6 in base_points:
                    continue
                
                # Check integer distances to P2, P3, P5
                if not is_int_dist(P6, P2):
                    continue
                if not is_int_dist(P6, P3):
                    continue
                if not is_int_dist(P6, P5):
                    continue
                
                # Check collinearity with all pairs
                valid = True
                for i in range(5):
                    for j in range(i+1, 5):
                        if are_collinear(P6, base_points[i], base_points[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check concyclicity with all triples
                for i in range(5):
                    for j in range(i+1, 5):
                        for k in range(j+1, 5):
                            if are_concyclic(P6, base_points[i], base_points[j], base_points[k]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        break
                    if not valid:
                        break
                
                if valid:
                    # Final check of the whole set
                    new_set = base_points + [P6]
                    if check_validity(new_set):
                        found_point = P6
                        break
            if found_point:
                break
        if found_point:
            break
            
    if found_point:
        final_points = base_points + [found_point]
    else:
        final_points = base_points
        
    # Convert to required format
    D = 1
    points_out = []
    for p in final_points:
        points_out.append([
            [p[0], 1],
            [p[1], 1]
        ])
        
    return {'D': D, 'points': points_out}
