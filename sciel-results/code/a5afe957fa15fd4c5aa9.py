"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _is_squarefree(n):
    if n <= 1:
        return True
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _make_point(x, y):
    # Returns [[x_num, x_den], [y_num, y_den]]
    if isinstance(x, Fraction):
        return [[x.numerator, x.denominator], [y.numerator, y.denominator]]
    return [[x, 1], [y, 1]]


def _check_pairwise_rational(points, D):
    # points: list of (Fraction x, Fraction y)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2.numerator // d2.denominator * d2.denominator):
                # Check if d2 is a perfect square rational
                # d2 is a Fraction; it is a perfect square rational iff num/den with num,den
                # squarefree parts... simpler: check if d2 is a square of a rational
                num = d2.numerator
                den = d2.denominator
                if not (_is_square(num) and _is_square(den)):
                    return False
    return True


def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Determinant = 0 if collinear
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False
    return True


def _check_no_four_concyclic(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points are concyclic iff the determinant of the circle equation is 0
                    # Circle: x^2 + D*y^2 + A*x + B*y + C = 0
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    
                    # The determinant of the 4x4 matrix:
                    # [x1^2 + D*y1^2, x1, y1, 1]
                    # [x2^2 + D*y2^2, x2, y2, 1]
                    # [x3^2 + D*y3^2, x3, y3, 1]
                    # [x4^2 + D*y4^2, x4, y4, 1]
                    # should be nonzero.
                    # To avoid huge numbers, we can translate to use the first point as origin.
                    # Subtract row1 from rows 2,3,4, then compute determinant of the resulting
                    # matrix. Actually, easier to just compute the 4x4 determinant directly
                    # but with Fractions it might be slow. Let's use a trick.
                    
                    # Translate so p1 is at origin:
                    # p1 -> (0,0)
                    # p2 -> (x2-x1, y2-y1)
                    # p3 -> (x3-x1, y3-y1)
                    # p4 -> (x4-x1, y4-y1)
                    # Then the circle through origin, p2', p3', p4' has equation:
                    # x^2 + D y^2 + A x + B y = 0
                    # So for p2': (x2'-x1)^2 + D(y2'-y1)^2 + A(x2'-x1) + B(y2'-y1) = 0
                    # Similarly for p3', p4'
                    # This gives a 2x3 system (2 unknowns A, B, 3 equations). 
                    # The system is consistent iff the 3x3 determinant (with columns for the 
                    # three points) is 0. Actually we have 3 equations, 2 unknowns, so the 
                    # system is consistent iff the 3x3 matrix [eq1, eq2, eq3] has rank 2.
                    
                    # Let's set up:
                    # For each of the 3 points relative to p1: (xi, yi)
                    # xi^2 + D*yi^2 + A*xi + B*yi = 0
                    # A*xi + B*yi = -(xi^2 + D*yi^2)
                    
                    # Matrix M (3x2) and vector v (3x1):
                    # M = [[x2', y2'], [x3', y3'], [x4', y4']]
                    # v = [-(x2'^2 + D*y2'^2), -(x3'^2 + D*y3'^2), -(x4'^2 + D*y4'^2)]
                    # Consistent iff the augmented 3x3 matrix has determinant 0.
                    
                    x2p = x2 - x1
                    y2p = y2 - y1
                    x3p = x3 - x1
                    y3p = y3 - y1
                    x4p = x4 - x1
                    y4p = y4 - y1
                    
                    # Determinant of:
                    # [x2p, y2p, -(x2p^2 + D*y2p^2)]
                    # [x3p, y3p, -(x3p^2 + D*y3p^2)]
                    # [x4p, y4p, -(x4p^2 + D*y4p^2)]
                    
                    # det = x2p*(y3p*(-(x4p^2+D*y4p^2)) - y4p*(-(x3p^2+D*y3p^2)))
                    #     - y2p*(x3p*(-(x4p^2+D*y4p^2)) - x4p*(-(x3p^2+D*y3p^2)))
                    #     + (-(x2p^2+D*y2p^2))*(x3p*y4p - x4p*y3p)
                    
                    det = (x2p * (y3p * (-(x4p**2 + D * y4p**2)) - y4p * (-(x3p**2 + D * y3p**2)))
                           - y2p * (x3p * (-(x4p**2 + D * y4p**2)) - x4p * (-(x3p**2 + D * y3p**2)))
                           + (-(x2p**2 + D * y2p**2)) * (x3p * y4p - x4p * y3p))
                    
                    if det == 0:
                        return False
    return True


def _validate(points_frac, D):
    if len(points_frac) < 3:
        return False
    if not _is_squarefree(D):
        return False
    if not _check_pairwise_rational(points_frac, D):
        return False
    if not _check_no_three_collinear(points_frac):
        return False
    if not _check_no_four_concyclic(points_frac, D):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 40s safety margin
    
    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    D = 1
    
    # Verify base
    if not _validate(base_frac, D):
        raise RuntimeError("Base configuration failed validation")
    
    best_points = base_frac
    best_count = len(base_frac)
    
    # Try to add points one at a time
    # Search space: rational points (x, y) with small numerators/denominators
    # We'll search in a bounded region around the existing points
    
    # For each candidate new point, check if adding it maintains all properties
    # We'll try integers first, then small fractions
    
    # Generate candidate points
    # The existing points have x in [-56, 16] and y in [0, 63]
    # Let's search in a slightly larger box
    
    search_range_x = range(-80, 40)
    search_range_y = range(-20, 80)
    
    # First pass: integer points
    for attempt in range(1, 200):
        if time.time() > deadline - 30:
            break
        
        found_new = False
        candidates = []
        
        # Generate random integer candidates
        import random
        random.seed(42 + attempt)
        
        for _ in range(200):
            x = random.randint(-80, 40)
            y = random.randint(-20, 80)
            candidates.append((Fraction(x), Fraction(y)))
        
        # Try each candidate
        for cand in candidates:
            if time.time() > deadline - 30:
                break
            
            new_points = best_points + [cand]
            if _validate(new_points, D):
                best_points = new_points
                best_count += 1
                found_new = True
                break
        
        if not found_new:
            # Try with larger search or different strategy
            # Maybe we should try to optimize the existing points instead
            # For now, if we can't add, we stop
            if best_count >= 6:
                break
            # Continue trying with different seeds
            continue
        else:
            # Continue to try to add more
            continue
    
    # Convert back to the required format
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}

# EVOLVE-BLOCK-END
