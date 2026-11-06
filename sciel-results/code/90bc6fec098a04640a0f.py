import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    # Base set of 5 points
    # These are known to be valid
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    
    def is_rational_sqrt(val):
        """Check if val is a perfect square of a rational number."""
        if val < 0:
            return None
        if val == 0:
            return Fraction(0)
        # val is a Fraction
        num, den = val.numerator, val.denominator
        # Check if num and den are perfect squares
        sqrt_num = int(math.isqrt(num))
        sqrt_den = int(math.isqrt(den))
        if sqrt_num * sqrt_num == num and sqrt_den * sqrt_den == den:
            return Fraction(sqrt_num, sqrt_den)
        return None

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def check_rational_distances(points):
        """Check if all pairwise distances are rational."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                d2 = dist_sq(points[i], points[j])
                if is_rational_sqrt(d2) is None:
                    return False
        return True

    def check_collinear(p1, p2, p3):
        """Check if p1, p2, p3 are collinear."""
        # Area of triangle is 0
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0

    def check_concyclic(p1, p2, p3, p4):
        """Check if p1, p2, p3, p4 are concyclic."""
        # Circle through p1, p2, p3:
        # x^2 + y^2 + Dx + Ey + F = 0
        # D, E, F can be found.
        # Or use determinant.
        # | x^2+y^2 x y 1 |
        # | x1^2+y1^2 x1 y1 1 |
        # | x2^2+y2^2 x2 y2 1 |
        # | x3^2+y3^2 x3 y3 1 |
        # det = 0 iff concyclic.
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
            
        def det4(m):
            # Expansion along first row
            return (m[0][0]*det3([[m[i][j] for j in [1,2,3]] for i in [1,2,3]]) -
                    m[0][1]*det3([[m[i][j] for j in [0,2,3]] for i in [1,2,3]]) +
                    m[0][2]*det3([[m[i][j] for j in [0,1,3]] for i in [1,2,3]]) -
                    m[0][3]*det3([[m[i][j] for j in [0,1,2]] for i in [1,2,3]]))
        
        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
        ]
        return det4(m) == 0

    def is_valid_set(points):
        if not check_rational_distances(points):
            return False
        n = len(points)
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def try_add_point(base_pts):
        """Try to add a 6th point to base_pts (which has 5 points)."""
        # Use P0 and P1 to define the circle intersection
        p0 = base_pts[0]
        p1 = base_pts[1]
        
        # Iterate over possible rational distances d0, d1
        # Limit search space
        max_num = 100
        for d0_num in range(1, max_num):
            for d0_den in range(1, max_num):
                d0 = Fraction(d0_num, d0_den)
                for d1_num in range(1, max_num):
                    for d1_den in range(1, max_num):
                        d1 = Fraction(d1_num, d1_den)
                        
                        # Check time
                        if time.time() > deadline:
                            return None
                        
                        # Find intersection of circles centered at p0, p1 with radii d0, d1
                        # (x-x0)^2 + (y-y0)^2 = d0^2
                        # (x-x1)^2 + (y-y1)^2 = d1^2
                        # Subtract:
                        # 2(x1-x0)x + 2(y1-y0)y = d0^2 - d1^2 + x1^2 - x0^2 + y1^2 - y0^2
                        # Let A = 2(x1-x0), B = 2(y1-y0), C = d0^2 - d1^2 + x1^2 - x0^2 + y1^2 - y0^2
                        # Ax + By = C
                        
                        x0, y0 = p0
                        x1, y1 = p1
                        A = 2 * (x1 - x0)
                        B = 2 * (y1 - y0)
                        C = d0*d0 - d1*d1 + x1*x1 - x0*x0 + y1*y1 - y0*y0
                        
                        if A == 0 and B == 0:
                            continue
                            
                        # If A != 0, x = (C - By)/A
                        # Substitute into circle 1:
                        # ((C - By)/A - x0)^2 + (y - y0)^2 = d0^2
                        # This is a quadratic in y: ay^2 + by + c = 0
                        # a = (B^2/A^2) + 1
                        # b = -2(B/A)(C/A - x0) - 2y0
                        # c = (C/A - x0)^2 + y0^2 - d0^2
                        
                        # To avoid fractions in coefficients, multiply by A^2:
                        # (B^2 + A^2)y^2 + B(-2(C - A x0) - 2 A y0) y + A^2((C/A - x0)^2 + y0^2 - d0^2) = 0
                        # Wait, (C/A - x0) = (C - A x0)/A
                        # So term is (C - A x0)^2 / A^2.
                        # Multiply by A^2:
                        # (A^2 + B^2)y^2 - 2B(C - A x0 + A y0)y + (C - A x0)^2 + A^2 y0^2 - A^2 d0^2 = 0
                        
                        a = A*A + B*B
                        b = -2 * B * (C - A*x0 + A*y0)
                        c = (C - A*x0)*(C - A*x0) + A*A*y0*y0 - A*A*d0*d0
                        
                        # Discriminant D = b^2 - 4ac
                        D = b*b - 4*a*c
                        sqrt_D = is_rational_sqrt(D)
                        if sqrt_D is None:
                            continue
                        
                        # y = (-b +/- sqrt_D) / (2a)
                        for sign in [1, -1]:
                            y = (-b + sign * sqrt_D) / (2 * a)
                            # x = (C - B y) / A
                            if A == 0:
                                continue
                            x = (C - B * y) / A
                            
                            new_point = (x, y)
                            # Check if distinct from existing
                            if any(new_point == p for p in base_pts):
                                continue
                                
                            # Check rational distances to all base points
                            valid = True
                            for p in base_pts:
                                if is_rational_sqrt(dist_sq(new_point, p)) is None:
                                    valid = False
                                    break
                            if not valid:
                                continue
                                
                            # Check general position
                            # Collinear with any pair
                            collinear = False
                            for i in range(len(base_pts)):
                                for j in range(i+1, len(base_pts)):
                                    if check_collinear(new_point, base_pts[i], base_pts[j]):
                                        collinear = True
                                        break
                                if collinear:
                                    break
                            if collinear:
                                continue
                                
                            # Concyclic with any triple
                            concyclic = False
                            for i in range(len(base_pts)):
                                for j in range(i+1, len(base_pts)):
                                    for k in range(j+1, len(base_pts)):
                                        if check_concyclic(new_point, base_pts[i], base_pts[j], base_pts[k]):
                                            concyclic = True
                                            break
                                    if concyclic:
                                        break
                                if concyclic:
                                    break
                            if concyclic:
                                continue
                                
                            return new_point
        return None

    # Try to add a 6th point
    new_point = try_add_point(base_points)
    if new_point:
        points = base_points + [new_point]
        # Final validation
        if is_valid_set(points):
            return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]}

    # Fallback to 5 points
    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base_points]}
