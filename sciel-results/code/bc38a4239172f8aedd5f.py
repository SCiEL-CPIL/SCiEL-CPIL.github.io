"""Construct integral point sets in general position."""
import time
import random
from math import isqrt, gcd
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n <= 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_rational_sqrt(numer, denom):
    """Check if numer/denom is a perfect square of a rational."""
    if numer < 0 or denom < 0:
        return False
    # Reduce
    g = gcd(numer, denom)
    if g > 0:
        numer //= g
        denom //= g
    rn = isqrt(numer)
    rd = isqrt(denom)
    return rn * rn == numer and rd * rd == denom

def _sq_dist(p1, p2, D):
    """Squared distance between two points (x1,y1),(x2,y2) in metric with parameter D."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _is_valid_dist(p1, p2, D):
    """Check if squared distance is a perfect square rational."""
    d2 = _sq_dist(p1, p2, D)
    if d2 <= 0:
        return False
    return _is_rational_sqrt(d2.numerator, d2.denominator)

def _no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(pts, D):
    """Check no four points lie on a 'circle' in the metric x^2 + D*y^2."""
    n = len(pts)
    for quad in combinations(range(n), 4):
        p1, p2, p3, p4 = (pts[i] for i in quad)
        # In the metric x^2 + D*y^2, a "circle" centered at (a,b) with radius r
        # satisfies (x-a)^2 + D*(y-b)^2 = r^2
        # Expanding: x^2 + D*y^2 - 2ax - 2Dby + (a^2 + D*b^2 - r^2) = 0
        # So the equation is: x^2 + D*y^2 + Ax + By + C = 0
        # where A = -2a, B = -2Db, C = a^2 + D*b^2 - r^2
        # Four points are concyclic iff the system:
        #   x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0 for i=1,2,3,4
        # has a solution (A,B,C), i.e., the 4x3 system is dependent.
        # Equivalently, the 4x4 determinant:
        # | x1  y1  x1^2+D*y1^2  1 |
        # | x2  y2  x2^2+D*y2^2  1 | = 0
        # | x3  y3  x3^2+D*y3^2  1 |
        # | x4  y4  x4^2+D*y4^2  1 |
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x, y, x * x + D * y * y, 1])
        # Compute 4x4 determinant
        a, b, c, d = rows[0]
        e, f, g, h = rows[1]
        i_, j_, k_, l_ = rows[2]
        m, n_, o, p_ = rows[3]
        det = (a * (f * (k_ * p_ - l_ * o) - g * (j_ * p_ - l_ * n_) + h * (j_ * o - k_ * n_))
               - b * (e * (k_ * p_ - l_ * o) - g * (i_ * p_ - l_ * m) + h * (i_ * o - k_ * m))
               + c * (e * (j_ * p_ - l_ * n_) - f * (i_ * p_ - l_ * m) + h * (i_ * n_ - j_ * m))
               - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
        if det == 0:
            return False
    return True

def _validate(points_xy, D):
    """Full validation."""
    n = len(points_xy)
    if n < 3:
        return False
    # Distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if points_xy[i][0] == points_xy[j][0] and points_xy[i][1] == points_xy[j][1]:
                return False
    # All pairwise distances rational
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_valid_dist(points_xy[i], points_xy[j], D):
                return False
    # No three collinear
    if not _no_three_collinear(points_xy):
        return False
    # No four concyclic
    if not _no_four_concyclic(points_xy, D):
        return False
    return True

def _find_6th_point_D1(base_points):
    """
    Attempt to find a 6th point for D=1 given 5 base points.
    Or find a 4th point given 3 base points (P1, P2, P3).
    This function specifically targets D=1 and uses the property that
    rational distances from (0,0) and (1,0) constrain the x-coordinate.
    """
    # This is a placeholder for a more targeted search if needed
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Known 5-point solution for D=1
    known_points = [(Fraction(0), Fraction(0)),
                    (Fraction(-56), Fraction(33)),
                    (Fraction(-16), Fraction(63)),
                    (Fraction(0), Fraction(33)),
                    (Fraction(16), Fraction(63))]
    
    best_points = known_points
    best_D = 1
    best_count = 5

    # Verify known solution
    if _validate(known_points, 1):
        pass  # confirmed valid
    else:
        # Fallback: try to find a valid 3-point set
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        best_D = 1
        best_count = 3

    random.seed(12345)

    # Strategy: Targeted search for 6 points in D=1
    # We use a construction based on rational points on circles.
    # P1 = (0,0), P2 = (1,0).
    # Find P3 such that d(P1,P3) and d(P2,P3) are rational.
    # Then find P4 such that d(Pi,P4) are rational for i=1,2,3.
    
    # Step 1: Find P3
    # Let P3 = (x, y).
    # x^2 + y^2 = a^2
    # (x-1)^2 + y^2 = b^2
    # 2x - 1 = a^2 - b^2
    # x = (a^2 - b^2 + 1) / 2
    # y^2 = a^2 - x^2
    # We search for small integer a, b such that y^2 is a rational square.
    
    p1 = (Fraction(0), Fraction(0))
    p2 = (Fraction(1), Fraction(0))
    p3_found = None
    
    for a in range(1, 100):
        if time.time() > deadline: break
        for b in range(1, 100):
            # a^2 - b^2 must be odd for x to be integer? No, x is rational.
            # x = (a^2 - b^2 + 1) / 2
            num_x = a*a - b*b + 1
            den_x = 2
            x = Fraction(num_x, den_x)
            y_sq = a*a - x*x
            if y_sq < 0: continue
            # Check if y_sq is a rational square
            if _is_rational_sqrt(y_sq.numerator, y_sq.denominator):
                y = Fraction(isqrt(y_sq.numerator), isqrt(y_sq.denominator))
                # Ensure y is not 0 (collinear with P1, P2)
                if y == 0: continue
                p3_candidate = (x, y)
                # Check distinctness
                if p3_candidate == p1 or p3_candidate == p2: continue
                p3_found = p3_candidate
                break
        if p3_found: break
    
    if p3_found:
        pts = [p1, p2, p3_found]
        # Step 2: Find P4
        # We need P4=(x,y) such that d(P4, Pi) are rational for i=1,2,3.
        # This is harder. We can iterate over possible distances c, d, e from P1, P2, P3.
        # c^2 = x^2 + y^2
        # d^2 = (x-1)^2 + y^2
        # e^2 = (x-x3)^2 + (y-y3)^2
        
        # From first two: x = (c^2 - d^2 + 1)/2
        # y^2 = c^2 - x^2
        # So for each pair (c,d), we get a candidate (x,y).
        # We then check if e^2 is a square.
        
        x3, y3 = p3_found
        p4_found = None
        
        for c in range(1, 50):
            if time.time() > deadline: break
            for d in range(1, 50):
                num_x = c*c - d*d + 1
                den_x = 2
                x = Fraction(num_x, den_x)
                y_sq = c*c - x*x
                if y_sq < 0: continue
                if not _is_rational_sqrt(y_sq.numerator, y_sq.denominator): continue
                y = Fraction(isqrt(y_sq.numerator), isqrt(y_sq.denominator))
                
                # Check distance to P3
                dx = x - x3
                dy = y - y3
                e_sq = dx*dx + dy*dy
                if e_sq <= 0: continue
                if _is_rational_sqrt(e_sq.numerator, e_sq.denominator):
                    p4_candidate = (x, y)
                    # Check distinctness and non-collinearity
                    if p4_candidate in pts: continue
                    # Check collinearity of P1, P2, P4 (y != 0)
                    if y == 0: continue
                    # Check collinearity of P1, P3, P4
                    cross1 = (p3_found[0]-p1[0])*(y-p1[1]) - (p3_found[1]-p1[1])*(x-p1[0])
                    if cross1 == 0: continue
                    # Check collinearity of P2, P3, P4
                    cross2 = (p3_found[0]-p2[0])*(y-p2[1]) - (p3_found[1]-p2[1])*(x-p2[0])
                    if cross2 == 0: continue
                    p4_found = p4_candidate
                    break
            if p4_found: break
        
        if p4_found:
            pts.append(p4_found)
            # Step 3: Find P5
            # Similar logic, but now we have 3 points.
            # We can try to find P5 by searching for distances to P1, P2, P3.
            # Or just use random search with the constraint that it must be at rational distance from P1, P2, P3.
            
            # Let's try a random search for P5, P6 constrained to be at rational distance from P1, P2.
            # Then check P3, P4.
            
            def _is_valid_candidate(p, existing_pts, D=1):
                for ep in existing_pts:
                    if not _is_valid_dist(p, ep, D):
                        return False
                return True

            p5_found = None
            # Search in a box around the existing points
            # Since coordinates can be large, we search in a moderate range first
            for _ in range(20000):
                if time.time() > deadline: break
                # Generate random rational x, y
                # To increase chance of rational distance from (0,0) and (1,0),
                # we can pick x from the form (c^2 - d^2 + 1)/2 for random c,d
                # and check if y^2 is a square.
                c = random.randint(1, 100)
                d = random.randint(1, 100)
                num_x = c*c - d*d + 1
                x = Fraction(num_x, 2)
                y_sq = c*c - x*x
                if y_sq < 0: continue
                if not _is_rational_sqrt(y_sq.numerator, y_sq.denominator): continue
                y = Fraction(isqrt(y_sq.numerator), isqrt(y_sq.denominator))
                
                p_candidate = (x, y)
                if p_candidate in pts: continue
                if y == 0: continue # Collinear with P1, P2
                
                # Check against P3, P4
                if _is_valid_candidate(p_candidate, pts):
                    # Check collinearity with all pairs
                    valid = True
                    for i in range(len(pts)):
                        for j in range(i+1, len(pts)):
                            p_a = pts[i]
                            p_b = pts[j]
                            cross = (p_candidate[0]-p_a[0])*(p_b[1]-p_a[1]) - (p_candidate[1]-p_a[1])*(p_b[0]-p_a[0])
                            if cross == 0:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        p5_found = p_candidate
                        break
            
            if p5_found:
                pts.append(p5_found)
                # Step 4: Find P6
                p6_found = None
                for _ in range(20000):
                    if time.time() > deadline: break
                    c = random.randint(1, 100)
                    d = random.randint(1, 100)
                    num_x = c*c - d*d + 1
                    x = Fraction(num_x, 2)
                    y_sq = c*c - x*x
                    if y_sq < 0: continue
                    if not _is_rational_sqrt(y_sq.numerator, y_sq.denominator): continue
                    y = Fraction(isqrt(y_sq.numerator), isqrt(y_sq.denominator))
                    
                    p_candidate = (x, y)
                    if p_candidate in pts: continue
                    if y == 0: continue
                    
                    if _is_valid_candidate(p_candidate, pts):
                        # Check collinearity
                        valid = True
                        for i in range(len(pts)):
                            for j in range(i+1, len(pts)):
                                p_a = pts[i]
                                p_b = pts[j]
                                cross = (p_candidate[0]-p_a[0])*(p_b[1]-p_a[1]) - (p_candidate[1]-p_a[1])*(p_b[0]-p_a[0])
                                if cross == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if valid:
                            # Check concyclicity
                            if _no_four_concyclic(pts + [p_candidate], 1):
                                p6_found = p_candidate
                                break
                
                if p6_found:
                    pts.append(p6_found)
                    best_points = pts
                    best_count = 6
                    # We found 6! Can we find 7?
                    # Unlikely in remaining time, but let's try a few more randoms
                    p7_found = None
                    for _ in range(5000):
                        if time.time() > deadline: break
                        c = random.randint(1, 100)
                        d = random.randint(1, 100)
                        num_x = c*c - d*d + 1
                        x = Fraction(num_x, 2)
                        y_sq = c*c - x*x
                        if y_sq < 0: continue
                        if not _is_rational_sqrt(y_sq.numerator, y_sq.denominator): continue
                        y = Fraction(isqrt(y_sq.numerator), isqrt(y_sq.denominator))
                        
                        p_candidate = (x, y)
                        if p_candidate in pts: continue
                        if y == 0: continue
                        
                        if _is_valid_candidate(p_candidate, pts):
                            valid = True
                            for i in range(len(pts)):
                                for j in range(i+1, len(pts)):
                                    p_a = pts[i]
                                    p_b = pts[j]
                                    cross = (p_candidate[0]-p_a[0])*(p_b[1]-p_a[1]) - (p_candidate[1]-p_a[1])*(p_b[0]-p_a[0])
                                    if cross == 0:
                                        valid = False
                                        break
                                if not valid: break
                            if valid:
                                if _no_four_concyclic(pts + [p_candidate], 1):
                                    p7_found = p_candidate
                                    break
                    if p7_found:
                        pts.append(p7_found)
                        best_points = pts
                        best_count = 7

    # Fallback to original random search if 6 not found
    if best_count < 6 and time.time() < deadline:
        # Original logic simplified
        pass

    # Convert to required format
    result_points = []
    for (x, y) in best_points:
        x_num = x.numerator
        x_den = x.denominator
        y_num = y.numerator
        y_den = y.denominator
        # Check bounds
        assert abs(x_num) <= 10**40 and x_den <= 10**40, f"x out of bounds: {x}"
        assert abs(y_num) <= 10**40 and y_den <= 10**40, f"y out of bounds: {y}"
        result_points.append([[x_num, x_den], [y_num, y_den]])

    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
