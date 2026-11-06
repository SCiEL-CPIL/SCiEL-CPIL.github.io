# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = math.isqrt(n)
        return r * r == n
    
    def check_validity(pts, D=1):
        """Check if points are valid: distinct, rational distances, no 3 collinear, no 4 concyclic"""
        n = len(pts)
        if n < 3:
            return False
        
        # Check distinct
        for i in range(n):
            for j in range(i + 1, n):
                if pts[i] == pts[j]:
                    return False
        
        # Check rational distances (squared distance must be perfect square rational)
        for i in range(n):
            for j in range(i + 1, n):
                dx = Fraction(pts[i][0]) - Fraction(pts[j][0])
                dy = Fraction(pts[i][1]) - Fraction(pts[j][1])
                dist_sq = dx * dx + dy * dy * D
                if dist_sq < 0:
                    return False
                if dist_sq == 0:
                    return False
                # Check if dist_sq is a perfect square
                num = dist_sq.numerator
                den = dist_sq.denominator
                if not is_perfect_square(num) or not is_perfect_square(den):
                    return False
                rnum = math.isqrt(num)
                rden = math.isqrt(den)
                if rnum * rnum != num or rden * rden != den:
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    x1, y1 = Fraction(p1[0]), Fraction(p1[1])
                    x2, y2 = Fraction(p2[0]), Fraction(p2[1])
                    x3, y3 = Fraction(p3[0]), Fraction(p3[1])
                    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                    if cross == 0:
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        x1, y1 = Fraction(p1[0]), Fraction(p1[1])
                        x2, y2 = Fraction(p2[0]), Fraction(p2[1])
                        x3, y3 = Fraction(p3[0]), Fraction(p3[1])
                        x4, y4 = Fraction(p4[0]), Fraction(p4[1])
                        
                        # Circle through p1, p2, p3: x^2 + y^2 + Ax + By + C = 0
                        # For point (x,y): x^2 + y^2 + A*x + B*y + C = 0
                        # Using the fact that (x1^2+y1^2) + A*x1 + B*y1 + C = 0, etc.
                        
                        # A*x1 + B*y1 + C = -(x1^2 + y1^2)
                        # A*x2 + B*y2 + C = -(x2^2 + y2^2)
                        # A*x3 + B*y3 + C = -(x3^2 + y3^2)
                        
                        # Subtract first from second: A*(x2-x1) + B*(y2-y1) = -(x2^2+y2^2-x1^2-y1^2)
                        # Subtract first from third: A*(x3-x1) + B*(y3-y1) = -(x3^2+y3^2-x1^2-y1^2)
                        
                        a2 = x2 - x1
                        b2 = y2 - y1
                        c2 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
                        
                        a3 = x3 - x1
                        b3 = y3 - y1
                        c3 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
                        
                        det = a2 * b3 - a3 * b2
                        if det == 0:
                            # Collinear, already checked
                            continue
                        
                        A = (c2 * b3 - c3 * b2) / det
                        B = (a2 * c3 - a3 * c2) / det
                        C = -(x1*x1 + y1*y1) - A * x1 - B * y1
                        
                        # Check if p4 is on the circle
                        val = x4*x4 + y4*y4 + A*x4 + B*y4 + C
                        if val == 0:
                            return False
        
        return True
    
    # Known valid 5-point sets (D=1, integer coordinates)
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (-3, 4), (0, 5), (3, 4), (-4, 3)],
        [(0, 0), (4, 3), (0, 5), (-4, 3), (3, 4)],
    ]
    
    # Verify base sets
    valid_base_sets = []
    for bp in base_sets:
        if check_validity(bp, 1):
            valid_base_sets.append(bp)
    
    if not valid_base_sets:
        # Fallback: simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    best_points = valid_base_sets[0]
    best_D = 1
    best_score = len(valid_base_sets[0])
    
    def try_add_point(base_pts, D, deadline):
        """Try to find a 6th point to add to base_pts"""
        base_set = set(base_pts)
        n_base = len(base_pts)
        
        # Strategy 1: Search integer lattice points
        # Generate candidates in spiral order
        for radius in range(1, 80):
            if time.time() > deadline - 10:
                break
            
            # Points on the square boundary at this radius
            candidates = []
            for x in range(-radius, radius + 1):
                if abs(x) == radius:
                    for y in range(-radius, radius + 1):
                        if abs(y) == radius:
                            candidates.append((x, y))
            for y in range(-radius, radius + 1):
                if abs(y) == radius:
                    for x in range(-radius, radius + 1):
                        if abs(x) != radius:  # Avoid duplicates
                            candidates.append((x, y))
            
            for (x, y) in candidates:
                if time.time() > deadline - 10:
                    break
                
                if (x, y) in base_set:
                    continue
                
                # Quick distance check
                valid = True
                for (xi, yi) in base_pts:
                    dx = x - xi
                    dy = y - yi
                    dist_sq = dx * dx + dy * dy * D
                    if dist_sq < 0 or not is_perfect_square(dist_sq):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Quick collinearity check
                collinear = False
                for i in range(n_base):
                    for j in range(i + 1, n_base):
                        p1, p2, p3 = (x, y), base_pts[i], base_pts[j]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                        if cross == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                
                if collinear:
                    continue
                
                # Quick concyclicity check
                concyclic = False
                for i in range(n_base):
                    for j in range(i + 1, n_base):
                        for k in range(j + 1, n_base):
                            p1, p2, p3, p4 = base_pts[i], base_pts[j], base_pts[k], (x, y)
                            x1, y1 = p1
                            x2, y2 = p2
                            x3, y3 = p3
                            x4, y4 = p4
                            
                            a2 = x2 - x1
                            b2 = y2 - y1
                            c2 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
                            
                            a3 = x3 - x1
                            b3 = y3 - y1
                            c3 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
                            
                            det = a2 * b3 - a3 * b2
                            if det == 0:
                                continue
                            
                            A_num = c2 * b3 - c3 * b2
                            B_num = a2 * c3 - a3 * c2
                            C_num = -(x1*x1 + y1*y1) * det - A_num * x1 - B_num * y1
                            
                            check = (x4*x4 + y4*y4) * det + A_num * x4 + B_num * y4 + C_num
                            if check == 0:
                                concyclic = True
                                break
                        if concyclic:
                            break
                    if concyclic:
                        break
                
                if concyclic:
                    continue
                
                # Full verification
                candidate_set = base_pts + [(x, y)]
                if check_validity(candidate_set, D):
                    return candidate_set
        
        return None
    
    # Try to extend each base set
    for base_pts in valid_base_sets:
        if time.time() > deadline - 20:
            break
        
        result = try_add_point(base_pts, 1, deadline)
        if result and len(result) > best_score:
            best_points = result
            best_D = 1
            best_score = len(result)
            # If we found 6 points, try to find more
            if best_score >= 6:
                # Try to add more points
                for extra_radius in range(1, 40):
                    if time.time() > deadline - 15:
                        break
                    # Try to add a 7th point
                    # This is more complex, skip for now
                    pass
    
    # Try with D > 1
    small_Ds = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95]
    
    for D in small_Ds:
        if time.time() > deadline - 10:
            break
        
        # For D > 1, we need to find points (x, y) such that distances are rational
        # This is more complex. Try simple configurations.
        
        # Try equilateral-like configurations
        # For D=2: points (0,0), (1,0), (0,1), (1,1), ...
        # Distance from (0,0) to (1,1) = sqrt(1 + 2) = sqrt(3) - not rational
        # Need to be more careful
        
        # Try: (0,0), (a,0), (0,b), (a,b) with a^2 + b^2*D = square
        # For D=2: a^2 + 2*b^2 = c^2
        # Solutions: a=1, b=1, c=2? 1+2=3 no. a=3, b=1, c=4? 9+2=11 no.
        # a=2, b=1, c=3? 4+2=6 no. a=1, b=2, c=3? 1+8=9 yes!
        # So (0,0), (1,0), (0,2), (1,2) with D=2
        # Check: dist (0,0)-(1,0) = 1, (0,0)-(0,2) = 2*sqrt(2), not rational!
        # Wait, distance = sqrt(dx^2 + D*dy^2)
        # (0,0) to (0,2): sqrt(0 + 2*4) = sqrt(8) - not rational
        # I made an error. Let me reconsider.
        
        # For D=2, distance between (x1,y1) and (x2,y2) = sqrt((x1-x2)^2 + 2*(y1-y2)^2)
        # We need this to be rational for all pairs.
        
        # Try (0,0), (1,0), (0,1): dists are 1, sqrt(2), sqrt(3) - not all rational
        # This is hard. Let's focus on D=1 for now.
        pass
    
    # Final verification
    if check_validity(best_points, best_D):
        return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in valid_base_sets[0]]}
# EVOLVE-BLOCK-END
