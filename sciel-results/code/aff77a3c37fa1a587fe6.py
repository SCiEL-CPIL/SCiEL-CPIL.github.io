"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from functools import reduce
from itertools import combinations

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_square(n):
    if n < 0:
        return False
    s = math.isqrt(n)
    return s * s == n

def is_squarefree(n):
    if n <= 0:
        return False
    n = abs(n)
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def make_squarefree(n):
    """Extract square factors from n, return (squarefree_part, square_factor_squared) such that n = sf * sf_sq^2."""
    if n == 0:
        return 0, 0
    n = abs(n)
    sf = 1
    sf_sq = 1
    p = 2
    while p * p <= n:
        while n % (p * p) == 0:
            n //= (p * p)
            sf_sq *= p
        p += 1
    sf = n
    return sf, sf_sq

# Points are represented as (x, y, D) where actual position is (x, y*sqrt(D)).
# x and y are Fractions. D is int (squarefree).

def dist_sq(p1, p2):
    """Return (num, den, D_effective) such that dist^2 = (num/den) + D_effective * ... wait no.
    dist^2 = (x1-x2)^2 + D*(y1-y2)^2.
    This is of the form A + B*D where A, B are rational.
    """
    x1, y1, D1 = p1
    x2, y2, D2 = p2
    # For points to be in the same field, we should have D1 == D2 in a standard setup.
    # Let's assume we are always working with a single D.
    if D1 != D2:
        # This shouldn't happen in our search strategy
        raise ValueError("Different D in distance calc")
    dx = x1 - x2
    dy = y1 - y2
    # dist_sq = dx^2 + D * dy^2
    # dx^2 is rational, D*dy^2 is D * rational.
    # We need dist_sq to be a perfect square of a rational number.
    # i.e. dist_sq = k^2 for some k in Q.
    val = dx * dx + D1 * dy * dy
    # val is a Fraction.
    return val

def dist_is_rational(p1, p2):
    d2 = dist_sq(p1, p2)
    return is_square(d2.numerator) and is_square(d2.denominator)

def collinear(p1, p2, p3):
    x1, y1, D = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    # det = (x2-x1)(y3-y1) - (x3-x1)(y2-y1) = 0
    # Note: y is coefficient of sqrt(D), so the geometry is in the plane.
    # Collinearity: (x2-x1)/(y2-y1) = (x3-x1)/(y3-y1) if y diffs non-zero.
    # More robust: (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

def concyclic(p1, p2, p3, p4):
    # Points are concyclic iff the determinant of the circle equation is 0.
    # x^2 + y^2*D + Dx + Ey + F = 0 is not the general form in this metric.
    # Actually, circles in the Euclidean plane defined by these coordinates.
    # Let points be (xi, yi).
    # General circle: (x-h)^2 + (y-k)^2 = r^2  => x^2 - 2hx + h^2 + y^2 - 2ky + k^2 = r^2
    # x^2 + y^2 - 2hx - 2ky + (h^2+k^2-r^2) = 0
    # Here "y^2" means (y*sqrt(D))^2 = y^2 D.
    # So: x^2 + D y^2 - 2h x - 2k y + C = 0
    # We check if there exist h, k, C such that this holds for 4 points.
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    # Equations:
    # x1^2 + D y1^2 - 2h x1 - 2k y1 + C = 0
    # x2^2 + D y2^2 - 2h x2 - 2k y2 + C = 0
    # x3^2 + D y3^2 - 2h x3 - 2k y3 + C = 0
    # x4^2 + D y4^2 - 2h x4 - 2k y4 + C = 0
    
    # This is a linear system for 2h, 2k, C.
    # Let A = 2h, B = 2k, C = C.
    # -x1 A - y1 B + C = -(x1^2 + D y1^2)
    # -x2 A - y2 B + C = -(x2^2 + D y2^2)
    # -x3 A - y3 B + C = -(x3^2 + D y3^2)
    # -x4 A - y4 B + C = -(x4^2 + D y4^2)
    
    # Subtract row 1 from rows 2, 3, 4:
    # -(x2-x1) A - (y2-y1) B = -(x2^2 - x1^2 + D(y2^2 - y1^2))
    # -(x3-x1) A - (y3-y1) B = -(x3^2 - x1^2 + D(y3^2 - y1^2))
    # -(x4-x1) A - (y4-y1) B = -(x4^2 - x1^2 + D(y4^2 - y1^2))
    
    # 3 equations, 2 unknowns (A, B).
    # For a solution to exist, the 3x3 determinant (or rank condition) must hold.
    # Specifically, the vector of RHS must be in the span of the columns of the coefficient matrix.
    
    x1, y1, D = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    x4, y4, _ = p4
    
    def rhs(p):
        return -(p[0]**2 + D * p[1]**2)
    
    r1 = rhs(p1)
    r2 = rhs(p2)
    r3 = rhs(p3)
    r4 = rhs(p4)
    
    # Coeff matrix M (3x2):
    # [-dx2 -dy2]
    # [-dx3 -dy3]
    # [-dx4 -dy4]
    
    # We can check if the 3x3 matrix formed by adding the RHS as a column has determinant 0.
    # Matrix:
    # [ -dx2  -dy2  r2-r1 ]
    # [ -dx3  -dy3  r3-r1 ]
    # [ -dx4  -dy4  r4-r1 ]
    
    a1 = -(x2 - x1)
    b1 = -(y2 - y1)
    c1 = r2 - r1
    
    a2 = -(x3 - x1)
    b2 = -(y3 - y1)
    c2 = r3 - r1
    
    a3 = -(x4 - x1)
    b3 = -(y4 - y1)
    c3 = r4 - r1
    
    # Det = a1(b2*c3 - c2*b3) - b1(a2*c3 - c2*a3) + c1(a2*b3 - b2*a3)
    det = a1 * (b2 * c3 - c2 * b3) - b1 * (a2 * c3 - c2 * a3) + c1 * (a2 * b3 - b2 * a3)
    return det == 0

def valid_set(points):
    n = len(points)
    # Check distinct
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    # Check rational distances
    for i in range(n):
        for j in range(i+1, n):
            if not dist_is_rational(points[i], points[j]):
                return False
    # Check no 3 collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if collinear(points[i], points[j], points[k]):
                    return False
    # Check no 4 concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def find_next_point(points, D, limit=10**4, timeout=5):
    """Search for a new point (x, y, D) such that dist to all existing points is rational."""
    start = time.time()
    
    # Convert all points to integers if possible for fast grid search
    # Check if all points have denominator 1
    all_int = all(p[0].denominator == 1 and p[1].denominator == 1 for p in points)
    
    if not all_int:
        # For non-integer points, we need a different approach
        # Try small rational points with denominator up to 3
        for den in [1, 2, 3]:
            if time.time() - start > timeout:
                return None
            for x_num in range(-200, 201):
                for y_num in range(-200, 201):
                    if time.time() - start > timeout:
                        return None
                    x = Fraction(x_num, den)
                    y = Fraction(y_num, den)
                    
                    # Check if this point is already in the set
                    skip = False
                    for p in points:
                        if p[0] == x and p[1] == y:
                            skip = True
                            break
                    if skip:
                        continue
                    
                    valid_so_far = True
                    for p in points:
                        d2 = (x - p[0])**2 + D * (y - p[1])**2
                        if not (is_square(d2.numerator) and is_square(d2.denominator)):
                            valid_so_far = False
                            break
                    
                    if not valid_so_far:
                        continue
                    
                    new_pt = (x, y, D)
                    
                    # Check collinearity with any pair of existing points
                    coll = False
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            if collinear(points[i], points[j], new_pt):
                                coll = True
                                break
                        if coll:
                            break
                    if coll:
                        continue
                    
                    # Check concyclicity with any triple of existing points
                    conc = False
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            for k in range(j+1, len(points)):
                                if concyclic(points[i], points[j], points[k], new_pt):
                                    conc = True
                                    break
                            if conc:
                                break
                        if conc:
                            break
                    if conc:
                        continue
                    
                    return new_pt
        return None
    
    # All points are integers - use fast integer grid search
    R = 100
    
    p0 = points[0]
    p0x = p0[0].numerator
    p0y = p0[1].numerator
    
    for x in range(-R, R + 1):
        if time.time() - start > timeout:
            return None
        for y in range(-R, R + 1):
            # Quick check: distance to p0 must be a perfect square
            d2_0 = (x - p0x)**2 + D * (y - p0y)**2
            if not is_square(d2_0):
                continue
            
            # Check all other points
            valid_so_far = True
            for p in points:
                px = p[0].numerator
                py = p[1].numerator
                d2 = (x - px)**2 + D * (y - py)**2
                if not is_square(d2):
                    valid_so_far = False
                    break
            
            if not valid_so_far:
                continue
            
            new_pt = (Fraction(x), Fraction(y), D)
            
            # Check collinearity with any pair of existing points
            coll = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if collinear(points[i], points[j], new_pt):
                        coll = True
                        break
                if coll:
                    break
            if coll:
                continue
            
            # Check concyclicity with any triple of existing points
            conc = False
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        if concyclic(points[i], points[j], points[k], new_pt):
                            conc = True
                            break
                    if conc:
                        break
                if conc:
                    break
            if conc:
                continue
            
            return new_pt
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    D = 1
    best_points = None
    best_count = 0
    
    # Try multiple seed sets
    seed_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 0), (0, 4), (5, 0), (0, 12)],
        [(0, 0), (100, 0), (150, 200), (260, 150), (400, 0)],
        [(0, 0), (65, 0), (0, 72), (104, 0), (0, 144)],
    ]
    
    for seed_pts in seed_sets:
        if time.time() > deadline:
            break
        
        points = [(Fraction(x), Fraction(y), D) for x, y in seed_pts]
        
        # Verify seed is valid
        if not valid_set(points):
            continue
        
        if len(points) > best_count:
            best_count = len(points)
            best_points = points[:]
        
        # Try to add points iteratively
        for _ in range(3):  # Try to add up to 3 more points
            if time.time() > deadline - 10:
                break
            remaining = max(1, int(deadline - time.time()) - 5)
            new_pt = find_next_point(points, D, timeout=remaining)
            if new_pt:
                points.append(new_pt)
                if valid_set(points):
                    if len(points) > best_count:
                        best_count = len(points)
                        best_points = points[:]
                else:
                    points.pop()
                    break
            else:
                break
    
    # Fallback to a simple valid set if nothing found
    if best_points is None:
        best_points = [(Fraction(0), Fraction(0), D), (Fraction(3), Fraction(0), D), (Fraction(0), Fraction(4), D)]
    
    # Convert back to output format
    final_points = []
    for p in best_points:
        x, y, _ = p
        final_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': final_points}

# EVOLVE-BLOCK-END
