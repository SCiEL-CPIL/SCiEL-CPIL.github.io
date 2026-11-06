# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Known valid 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verification function
    def is_valid(pts):
        n = len(pts)
        # Check distinct
        for i in range(n):
            for j in range(i+1, n):
                if pts[i] == pts[j]:
                    return False
        
        # Check rational distances
        for i in range(n):
            for j in range(i+1, n):
                dx = Fraction(pts[i][0]) - Fraction(pts[j][0])
                dy = Fraction(pts[i][1]) - Fraction(pts[j][1])
                dist_sq = dx*dx + dy*dy
                if dist_sq.denominator != 1:
                    return False
                num = dist_sq.numerator
                if num < 0:
                    return False
                root = int(num**0.5)
                if root*root != num:
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0]):
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        dx2 = Fraction(x2-x1)
                        dy2 = Fraction(y2-y1)
                        rhs2 = Fraction(-(x2*x2+y2*y2 - x1*x1-y1*y1))
                        
                        dx3 = Fraction(x3-x1)
                        dy3 = Fraction(y3-y1)
                        rhs3 = Fraction(-(x3*x3+y3*y3 - x1*x1-y1*y1))
                        
                        det = dx2*dy3 - dx3*dy2
                        if det == 0:
                            return False
                            
                        A = (rhs2*dy3 - rhs3*dy2) / det
                        B = (dx2*rhs3 - dx3*rhs2) / det
                        C = Fraction(-(x1*x1+y1*y1)) - A*x1 - B*y1
                        
                        val = x4*x4 + y4*y4 + A*x4 + B*y4 + C
                        if val == 0:
                            return False
                            
        return True

    # First verify the base set is valid
    if not is_valid(base_points):
        # Fallback: return a simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    # Try to find a 6th point
    # The 6th point (x, y) must satisfy:
    # 1. (x-xi)^2 + (y-yi)^2 is a perfect square for all i in 0..4
    # 2. No three collinear
    # 3. No four concyclic
    
    # Strategy: Search for integer (x, y) in a reasonable range
    # For each candidate, check if distances to all 5 base points are rational
    
    best_points = base_points  # Default to 5 points
    
    # Try various ranges
    search_ranges = [
        (-100, 100),
        (-200, 200),
        (-300, 300),
    ]
    
    found_sixth = False
    for xmin, xmax in search_ranges:
        if time.time() > deadline - 10:
            break
        for x in range(xmin, xmax + 1):
            if time.time() > deadline - 10:
                break
            for y in range(xmin, xmax + 1):
                if time.time() > deadline - 10:
                    break
                
                # Skip if this point is already in base_points
                if (x, y) in base_points:
                    continue
                
                # Quick check: distances to all 5 base points must be rational
                valid_candidate = True
                for (xi, yi) in base_points:
                    dx = x - xi
                    dy = y - yi
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 0:
                        valid_candidate = False
                        break
                    root = int(math.isqrt(dist_sq))
                    if root * root != dist_sq:
                        valid_candidate = False
                        break
                
                if not valid_candidate:
                    continue
                
                # Check collinearity with existing points
                collinear_found = False
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        p1 = (x, y)
                        p2 = base_points[i]
                        p3 = base_points[j]
                        if (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0]):
                            collinear_found = True
                            break
                    if collinear_found:
                        break
                
                if collinear_found:
                    continue
                
                # Check concyclicity: no 4 points (including new) should be concyclic
                # This means: no 3 base points + new point should be concyclic
                concyclic_found = False
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        for k in range(j+1, len(base_points)):
                            p1 = base_points[i]
                            p2 = base_points[j]
                            p3 = base_points[k]
                            p4 = (x, y)
                            
                            x1, y1 = p1
                            x2, y2 = p2
                            x3, y3 = p3
                            x4, y4 = p4
                            
                            dx2 = x2 - x1
                            dy2 = y2 - y1
                            rhs2 = -(x2*x2+y2*y2 - x1*x1-y1*y1)
                            
                            dx3 = x3 - x1
                            dy3 = y3 - y1
                            rhs3 = -(x3*x3+y3*y3 - x1*x1-y1*y1)
                            
                            det = dx2*dy3 - dx3*dy2
                            if det == 0:
                                # Collinear, skip (already checked)
                                continue
                                
                            # Check if p4 lies on circle through p1,p2,p3
                            # A*x + B*y + C = -(x^2+y^2)
                            # Using Cramer's rule with integers
                            # det_A = rhs2*dy3 - rhs3*dy2
                            # det_B = dx2*rhs3 - dx3*rhs2
                            # det_C = dx2*dy3*rhs1 - dx3*dy2*rhs1 ... wait, need to be more careful
                            
                            # Actually, let's use the determinant method directly
                            # 4 points concyclic iff det of this 4x4 matrix is 0:
                            # | x1 y1 x1^2+y1^2 1 |
                            # | x2 y2 x2^2+y2^2 1 |
                            # | x3 y3 x3^2+y3^2 1 |
                            # | x4 y4 x4^2+y4^2 1 |
                            
                            # Compute determinant using expansion
                            # Let's compute it step by step
                            a1 = x1; b1 = y1; c1 = x1*x1+y1*y1
                            a2 = x2; b2 = y2; c2 = x2*x2+y2*y2
                            a3 = x3; b3 = y3; c3 = x3*x3+y3*y3
                            a4 = x4; b4 = y4; c4 = x4*x4+y4*y4
                            
                            # det = a1*(b2*(c3*1 - 1*c4) - b3*(c2*1 - 1*c4) + c3*(b2*1 - b3*1))
                            #     - b1*(a2*(c3*1 - 1*c4) - a3*(c2*1 - 1*c4) + c2*(a2*1 - a3*1))
                            #     + c1*(a2*(b3*1 - 1*b4) - a3*(b2*1 - 1*b4) + b2*(a3*1 - a2*1))
                            #     - 1*(a2*(b3*c4 - c3*b4) - a3*(b2*c4 - c2*b4) + c2*(b2*a3 - a2*b3))
                            
                            # This is getting complex. Let's use a simpler check:
                            # Compute circle through p1,p2,p3 and check if p4 is on it.
                            # Using the formula:
                            # A = (rhs2*dy3 - rhs3*dy2) / det
                            # B = (dx2*rhs3 - dx3*rhs2) / det
                            # C = -(x1^2+y1^2) - A*x1 - B*y1
                            # Check: x4^2+y4^2 + A*x4 + B*y4 + C == 0
                            # i.e., (x4^2+y4^2)*det + A_num*x4 + B_num*y4 + C_num == 0
                            # where A_num = rhs2*dy3 - rhs3*dy2
                            #       B_num = dx2*rhs3 - dx3*rhs2
                            #       C_num = -(x1^2+y1^2)*det - A_num*x1 - B_num*y1
                            
                            A_num = rhs2*dy3 - rhs3*dy2
                            B_num = dx2*rhs3 - dx3*rhs2
                            C_num = -(x1*x1+y1*y1)*det - A_num*x1 - B_num*y1
                            
                            check = (x4*x4+y4*y4)*det + A_num*x4 + B_num*y4 + C_num
                            if check == 0:
                                concyclic_found = True
                                break
                        if concyclic_found:
                            break
                    if concyclic_found:
                        break
                
                if concyclic_found:
                    continue
                
                # All checks passed! We found a valid 6th point
                best_points = base_points + [(x, y)]
                found_sixth = True
                break
            if found_sixth:
                break
        
        if found_sixth:
            break
    
    # Final verification
    if is_valid(best_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points]}
# EVOLVE-BLOCK-END
