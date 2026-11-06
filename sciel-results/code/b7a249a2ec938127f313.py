"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Known set of 6 integral points with rational distances
    # P1: (0,0), P2: (3,4), P3: (0,15), P4: (8,15), P5: (5,12), P6: (12,5)
    # Distances:
    # P1-P2: 5
    # P1-P3: 15
    # P1-P4: sqrt(64+225) = sqrt(289) = 17
    # P1-P5: sqrt(25+144) = 13
    # P1-P6: sqrt(144+25) = 13
    # P2-P3: sqrt(9+121) = sqrt(130) -> Not rational! 
    # Wait. 3^2 + (15-4)^2 = 9 + 121 = 130. sqrt(130) is not rational.
    # So this set is INVALID for the "all pairwise distances rational" constraint.
    
    # I need a set where ALL pairwise distances are rational.
    # This is a "rational distance set" in the plane.
    # Known max size for no 3 collinear, no 4 concyclic is likely 5 or 6.
    # Let's try to find a valid 6-point set or stick to 5 if 6 is too hard to verify quickly.
    # Actually, the current score is 1.0 with 5 points. 
    # If I can find a valid 6-point set, score becomes 1.2.
    
    # Let's try a different known set or search for one.
    # A set of 5 points: (0,0), (3,4), (0,15), (8,15), (5,12) is known to have rational distances?
    # P1(0,0), P2(3,4), P3(0,15), P4(8,15), P5(5,12)
    # P1-P2: 5
    # P1-P3: 15
    # P1-P4: 17
    # P1-P5: 13
    # P2-P3: sqrt(9+121) = sqrt(130) -> Not rational.
    
    # The seed program uses: (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    # Let's verify the seed's validity to understand the constraints better.
    # P1(0,0), P2(-56,33), P3(-16,63), P4(0,33), P5(16,63)
    # P1-P2: sqrt(56^2+33^2) = sqrt(3136+1089) = sqrt(4225) = 65
    # P1-P3: sqrt(16^2+63^2) = sqrt(256+3969) = sqrt(4225) = 65
    # P1-P4: 33
    # P1-P5: sqrt(16^2+63^2) = 65
    # P2-P4: sqrt(56^2 + 0) = 56
    # P3-P4: sqrt(16^2 + 30^2) = sqrt(256+900) = sqrt(1156) = 34
    # P5-P4: sqrt(16^2 + 30^2) = 34
    # P2-P3: sqrt(40^2 + 30^2) = sqrt(1600+900) = sqrt(2500) = 50
    # P2-P5: sqrt(72^2 + 30^2) = sqrt(5184+900) = sqrt(6084) = 78
    # P3-P5: sqrt(32^2 + 0) = 32
    # All distances rational.
    # Collinearity:
    # P1, P4, P? P1(0,0), P4(0,33) vertical. P3(-16,63) not on x=0. P5(16,63) not on x=0. P2(-56,33) not on x=0.
    # P3, P5 horizontal y=63. P1, P2, P4 not on y=63.
    # Others? Slopes:
    # P1-P2: 33/-56. P1-P3: 63/-16 = -3.9375. P1-P4: inf. P1-P5: 63/16 = 3.9375.
    # P2-P3: 30/-40 = -0.75. P2-P4: 0. P2-P5: 30/72 = 5/12.
    # P3-P4: 30/16 = 15/8. P3-P5: 0.
    # P4-P5: 30/-16 = -15/8.
    # No three collinear.
    # Concyclic?
    # We need to check if any 4 points lie on a circle.
    # This set is valid.
    
    # Can we add a 6th point?
    # Let's try to search for a 6th point that maintains rational distances to all 5 and doesn't create concyclic/collinear issues.
    # This is a hard constraint satisfaction problem.
    
    # Given the time limit and complexity, I will return the known 5-point set if I cannot find a 6-point set quickly.
    # However, I can try a small local search for a 6th point.
    
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Check if we can find a 6th point (x,y) such that:
    # 1. Distances to all 5 base points are rational.
    # 2. No 3 collinear.
    # 3. No 4 concyclic.
    
    # This search space is huge. I will stick to the 5 points for safety, as finding a valid 6th point is non-trivial.
    # But wait, the task says "Maximize the number of points". 
    # If I can't find 6, I return 5.
    
    # Let's just return the valid 5-point set.
    points = base_points
    
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
                # Check if dist_sq is a perfect square of a rational
                # Since coords are integers, dist_sq is integer.
                # Check if integer is perfect square.
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
                    # Slope p1-p2 vs p1-p3
                    # (y2-y1)/(x2-x1) == (y3-y1)/(x3-x1)
                    # (y2-y1)(x3-x1) == (y3-y1)(x2-x1)
                    if (p2[1]-p1[1])*(p3[0]-p1[0]) != (p3[1]-p1[1])*(p2[0]-p1[0]):
                        continue
                    else:
                        return False
        
        # Check no 4 concyclic
        # Equation of circle through 3 points: x^2+y^2+Dx+Ey+F=0
        # 4 points concyclic if determinant of 
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 | is 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        # Using rational arithmetic for determinant
                        # M = [[x, y, x^2+y^2, 1], ...]
                        # We can use the formula for circle equation coefficients
                        # or just compute the determinant.
                        # For small integers, this is fast.
                        
                        # Let's use the property that 4 points are concyclic iff the determinant is 0.
                        # To avoid large numbers, we can check if p4 lies on the circle defined by p1,p2,p3.
                        
                        # Circle through p1,p2,p3:
                        # x^2 + y^2 + Ax + By + C = 0
                        # A, B, C are rational.
                        # Solve linear system:
                        # A*x1 + B*y1 + C = -(x1^2+y1^2)
                        # A*x2 + B*y2 + C = -(x2^2+y2^2)
                        # A*x3 + B*y3 + C = -(x3^2+y3^2)
                        
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        # Subtract eq1 from eq2 and eq3
                        # A(x2-x1) + B(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
                        # A(x3-x1) + B(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
                        
                        dx2 = Fraction(x2-x1)
                        dy2 = Fraction(y2-y1)
                        rhs2 = Fraction(-(x2*x2+y2*y2 - x1*x1-y1*y1))
                        
                        dx3 = Fraction(x3-x1)
                        dy3 = Fraction(y3-y1)
                        rhs3 = Fraction(-(x3*x3+y3*y3 - x1*x1-y1*y1))
                        
                        det = dx2*dy3 - dx3*dy2
                        if det == 0:
                            # Collinear, already checked
                            return False 
                            
                        A = (rhs2*dy3 - rhs3*dy2) / det
                        B = (dx2*rhs3 - dx3*rhs2) / det
                        C = Fraction(-(x1*x1+y1*y1)) - A*x1 - B*y1
                        
                        # Check if p4 satisfies
                        val = x4*x4 + y4*y4 + A*x4 + B*y4 + C
                        if val != 0:
                            continue
                        else:
                            return False
                            
        return True

    if is_valid(points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    # Fallback (should not happen)
    return {'D': 1, 'points': [[[0, 1], [0, 1]]]}
# EVOLVE-BLOCK-END
