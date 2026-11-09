import math
import time
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Try to find a 6-point set using Q(sqrt(3))
    # Points are (x, y*sqrt(3)) where x, y are rational
    # Start with a regular hexagon in this field
    
    def check_valid(points, D):
        """Check all constraints with exact arithmetic.
        points: list of (x_num, x_den, y_num, y_den) representing (x, y*sqrt(D))
        """
        n = len(points)
        
        # Check distinctness
        for i in range(n):
            for j in range(i+1, n):
                xi = Fraction(points[i][0], points[i][1])
                yi = Fraction(points[i][2], points[i][3])
                xj = Fraction(points[j][0], points[j][1])
                yj = Fraction(points[j][2], points[j][3])
                if xi == xj and yi == yj:
                    return False
        
        # Check all pairwise distances are rational
        # dist^2 = (x1-x2)^2 + D*(y1-y2)^2 must be a perfect square of a rational
        for i in range(n):
            for j in range(i+1, n):
                xi = Fraction(points[i][0], points[i][1])
                yi = Fraction(points[i][2], points[i][3])
                xj = Fraction(points[j][0], points[j][1])
                yj = Fraction(points[j][2], points[j][3])
                dx = xi - xj
                dy = yi - yj
                d2 = dx*dx + D * dy*dy
                if d2 < 0:
                    return False
                # Check if d2 is a perfect square of a rational
                # d2 = p/q in lowest terms, need p and q to be perfect squares
                if d2 == 0:
                    continue
                num = d2.numerator
                den = d2.denominator
                s_num = int(math.isqrt(num))
                s_den = int(math.isqrt(den))
                if s_num * s_num != num or s_den * s_den != den:
                    return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    xi = Fraction(points[i][0], points[i][1])
                    yi = Fraction(points[i][2], points[i][3])
                    xj = Fraction(points[j][0], points[j][1])
                    yj = Fraction(points[j][2], points[j][3])
                    xk = Fraction(points[k][0], points[k][1])
                    yk = Fraction(points[k][2], points[k][3])
                    # Cross product should be nonzero
                    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                    if cross == 0:
                        return False
        
        # Check no four concyclic
        # Four points are concyclic iff the determinant of the 4x4 matrix is zero
        # | x^2 + D*y^2  x  y  1 |
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if time.time() > deadline:
                            return False  # timeout, treat as invalid for safety
                        pts = [points[i], points[j], points[k], points[l]]
                        # Compute determinant of:
                        # | x^2 + D*y^2  x  y  1 |
                        # | x^2 + D*y^2  x  y  1 |
                        # | x^2 + D*y^2  x  y  1 |
                        # | x^2 + D*y^2  x  y  1 |
                        rows = []
                        for p in pts:
                            x = Fraction(p[0], p[1])
                            y = Fraction(p[2], p[3])
                            x2dy2 = x*x + D*y*y
                            rows.append([x2dy2, x, y, 1])
                        # Compute 4x4 determinant
                        # Using cofactor expansion
                        def det4(m):
                            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2]
                                    - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1])
                                    - m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2]
                                    - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0])
                                    + m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1]
                                    - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0])
                                    - m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1]
                                    - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
                        if det4(rows) == 0:
                            return False
        
        return True
    
    # Strategy 1: Try Q(sqrt(3)) with regular hexagon + perturbation
    # Regular hexagon vertices in Q(sqrt(3)):
    # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # In our representation (x, y*sqrt(3)):
    # (1, 0), (1/2, 1/2), (-1/2, 1/2), (-1, 0), (-1/2, -1/2), (1/2, -1/2)
    
    D = 3
    hex_points = [
        (1, 1, 0, 1),   # (1, 0)
        (1, 2, 1, 2),   # (1/2, sqrt(3)/2)
        (-1, 2, 1, 2),  # (-1/2, sqrt(3)/2)
        (-1, 1, 0, 1),  # (-1, 0)
        (-1, 2, -1, 2), # (-1/2, -sqrt(3)/2)
        (1, 2, -1, 2),  # (1/2, -sqrt(3)/2)
    ]
    
    # Check if hexagon is valid (it shouldn't be - all 6 are concyclic)
    hex_valid = check_valid(hex_points, D)
    if hex_valid:
        return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in hex_points]}
    
    # The hexagon has all 6 points on a circle. We need to break this.
    # Try perturbing one point slightly.
    # Point 5: (1/2, -1/2) -> (1/2 + a, -1/2 + b) for small rational a, b
    
    # For the perturbed point to have rational distance to all others,
    # we need specific conditions. Let's search over small perturbations.
    
    best_6 = None
    
    # Try perturbing different points
    for perturb_idx in range(6):
        if time.time() > deadline:
            break
        
        base_points = hex_points[:]
        
        # Try small rational perturbations to point perturb_idx
        # (x0 + a/b, y0 + c/d) where a,b,c,d are small
        x0_num, x0_den = base_points[perturb_idx][0], base_points[perturb_idx][1]
        y0_num, y0_den = base_points[perturb_idx][2], base_points[perturb_idx][3]
        
        x0 = Fraction(x0_num, x0_den)
        y0 = Fraction(y0_num, y0_den)
        
        # Try perturbations a/b + i*c/d where |a|,|b|,|c|,|d| <= small
        max_pert = 50
        for a in range(-max_pert, max_pert + 1):
            if time.time() > deadline:
                break
            for b in range(1, max_pert + 1):
                if time.time() > deadline:
                    break
                dx = Fraction(a, b)
                for c in range(-max_pert, max_pert + 1):
                    if time.time() > deadline:
                        break
                    for d in range(1, max_pert + 1):
                        if time.time() > deadline:
                            break
                        dy = Fraction(c, d)
                        
                        new_x = x0 + dx
                        new_y = y0 + dy
                        
                        # Skip if too large or same as original
                        if abs(new_x.numerator) > 10**10 or abs(new_x.denominator) > 10**10:
                            continue
                        if abs(new_y.numerator) > 10**10 or abs(new_y.denominator) > 10**10:
                            continue
                        if new_x == x0 and new_y == y0:
                            continue
                        
                        # Check if new point coincides with any other
                        new_pt = (new_x.numerator, new_x.denominator, new_y.numerator, new_y.denominator)
                        # Normalize
                        from math import gcd
                        gx = gcd(new_x.numerator, new_x.denominator)
                        gy = gcd(new_y.numerator, new_y.denominator)
                        new_pt = (new_x.numerator//gx, new_x.denominator//gx, 
                                  new_y.numerator//gy, new_y.denominator//gy)
                        
                        coincides = False
                        for i in range(6):
                            if i == perturb_idx:
                                continue
                            xi = Fraction(hex_points[i][0], hex_points[i][1])
                            yi = Fraction(hex_points[i][2], hex_points[i][3])
                            if new_x == xi and new_y == yi:
                                coincides = True
                                break
                        if coincides:
                            continue
                        
                        # Build candidate set
                        candidate = hex_points[:]
                        candidate[perturb_idx] = new_pt
                        
                        # Quick check: distances to all other points
                        valid_dist = True
                        for i in range(6):
                            if i == perturb_idx:
                                continue
                            xi = Fraction(candidate[i][0], candidate[i][1])
                            yi = Fraction(candidate[i][2], candidate[i][3])
                            dx2 = new_x - xi
                            dy2 = new_y - yi
                            d2 = dx2*dx2 + D * dy2*dy2
                            if d2 < 0:
                                valid_dist = False
                                break
                            if d2 == 0:
                                valid_dist = False
                                break
                            num = d2.numerator
                            den = d2.denominator
                            s_num = int(math.isqrt(num))
                            s_den = int(math.isqrt(den))
                            if s_num * s_num != num or s_den * s_den != den:
                                valid_dist = False
                                break
                        if not valid_dist:
                            continue
                        
                        # Check no three collinear
                        valid_collinear = True
                        for i in range(6):
                            for j in range(i+1, 6):
                                for k in range(j+1, 6):
                                    xi = Fraction(candidate[i][0], candidate[i][1])
                                    yi = Fraction(candidate[i][2], candidate[i][3])
                                    xj = Fraction(candidate[j][0], candidate[j][1])
                                    yj = Fraction(candidate[j][2], candidate[j][3])
                                    xk = Fraction(candidate[k][0], candidate[k][1])
                                    yk = Fraction(candidate[k][2], candidate[k][3])
                                    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                                    if cross == 0:
                                        valid_collinear = False
                                        break
                                if not valid_collinear:
                                    break
                            if not valid_collinear:
                                break
                        if not valid_collinear:
                            continue
                        
                        # Check no four concyclic
                        valid_concyclic = True
                        for i in range(6):
                            for j in range(i+1, 6):
                                for k in range(j+1, 6):
                                    for l in range(k+1, 6):
                                        if time.time() > deadline:
                                            valid_concyclic = False
                                            break
                                        pts = [candidate[i], candidate[j], candidate[k], candidate[l]]
                                        rows = []
                                        for p in pts:
                                            x = Fraction(p[0], p[1])
                                            y = Fraction(p[2], p[3])
                                            x2dy2 = x*x + D*y*y
                                            rows.append([x2dy2, x, y, 1])
                                        
                                        def det4(m):
                                            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2]
                                                    - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1])
                                                    - m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2]
                                                    - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0])
                                                    + m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1]
                                                    - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0])
                                                    - m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1]
                                                    - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
                                        
                                        if det4(rows) == 0:
                                            valid_concyclic = False
                                            break
                                    if not valid_concyclic:
                                        break
                                if not valid_concyclic:
                                    break
                            if not valid_concyclic:
                                break
                        if not valid_concyclic:
                            continue
                        
                        # Found a valid 6-point set!
                        best_6 = candidate
                        break
                        if time.time() > deadline:
                            break
                    if best_6:
                        break
                if best_6:
                    break
            if best_6:
                break
        if best_6:
            break
    
    if best_6:
        # Verify with full check
        if check_valid(best_6, D):
            return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in best_6]}
    
    # Fallback: use the known 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}

