"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from math import isqrt, gcd, prod
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n <= 1:
        return True
    n = abs(n)
    if n % 4 == 0:
        return False
    # Check small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n > 1:
        if n % 4 == 0:
            return False
        # For remaining n, check if it's a square
        r = isqrt(n)
        if r * r == n:
            return False
    return True

def _dist_sq(p1, p2, D):
    """Compute squared distance between p1 and p2 where points are (x, y*sqrt(D)).
    Returns a Fraction.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    # dist^2 = dx^2 + D * dy^2
    return dx * dx + D * dy * dy

def _is_rational_dist(p1, p2, D):
    """Check if distance between p1 and p2 is rational."""
    d2 = _dist_sq(p1, p2, D)
    return _is_square(d2.numerator) and _is_square(d2.denominator)

def _collinear(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # Cross product
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic using exact arithmetic."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Circle through p1, p2, p3 has equation x^2 + y^2 + Dx + Ey + F = 0
    # where y is actually y*sqrt(D), so x^2 + D*y^2 + Dx + Ey*sqrt(D) + F = 0
    # Actually, let's work in the field Q(sqrt(D)).
    # A circle in this geometry: x^2 + D*y^2 + ax + by*sqrt(D) + c = 0
    # Wait, the metric is dx^2 + D*dy^2. So a "circle" is x^2 + D*y^2 + ax + by + c = 0?
    # No. The distance squared is (x1-x2)^2 + D*(y1-y2)^2.
    # A circle centered at (xc, yc) with radius r: (x-xc)^2 + D*(y-yc)^2 = r^2
    # x^2 - 2xc*x + xc^2 + D*y^2 - 2D*yc*y + D*yc^2 = r^2
    # x^2 + D*y^2 - 2xc*x - 2D*yc*y + (xc^2 + D*yc^2 - r^2) = 0
    # So the general circle equation is x^2 + D*y^2 + ax + by + c = 0
    # where a, b, c are rational.
    
    # For point (xi, yi) to be on the circle:
    # xi^2 + D*yi^2 + a*xi + b*yi + c = 0
    
    # We have 3 unknowns a, b, c and 3 equations from p1, p2, p3.
    # Then check if p4 satisfies the equation.
    
    # Equation for pi:
    # a*xi + b*yi + c = -(xi^2 + D*yi^2)
    
    # Let's solve the linear system for a, b, c.
    # Matrix:
    # [x1, y1, 1] [a]   [-(x1^2 + D*y1^2)]
    # [x2, y2, 1] [b] = [-(x2^2 + D*y2^2)]
    # [x3, y3, 1] [c]   [-(x3^2 + D*y3^2)]
    
    # Determinant of the matrix:
    det = (x1*(y2 - y3) - y1*(x2 - x3) + (x2*y3 - x3*y2))
    
    if det == 0:
        # Points are collinear, no unique circle
        return False
    
    # Cramer's rule or direct substitution
    # a = ( (-x1^2-D*y1^2)*(y2-y3) - y1*(-x2^2-D*y2^2 + x3^2+D*y3^2) + (x2*(-x3^2-D*y3^2) - x3*(-x2^2-D*y2^2)) ) / det
    # This is getting messy. Let's use a different approach.
    
    # The condition for 4 points to be concyclic is that the cross ratio is real,
    # or we can use the determinant of a 4x4 matrix.
    # For points (xi, yi) in a metric with squared distance dx^2 + D*dy^2,
    # the condition is:
    # | x1^2 + D*y1^2, x1, y1, 1 |
    # | x2^2 + D*y2^2, x2, y2, 1 | = 0
    # | x3^2 + D*y3^2, x3, y3, 1 |
    # | x4^2 + D*y4^2, x4, y4, 1 |
    
    # Let's compute this determinant.
    A = x1*x1 + D*y1*y1
    B = x2*x2 + D*y2*y2
    C = x3*x3 + D*y3*y3
    E = x4*x4 + D*y4*y4
    
    # Det = A*(x2*(y3-y4) - y2*(x3-x4) + (x3*y4 - x4*y3))
    #     - B*(x1*(y3-y4) - y1*(x3-x4) + (x3*y4 - x4*y3))
    #     + C*(x1*(y2-y4) - y1*(x2-x4) + (x2*y4 - x4*y2))
    #     - E*(x1*(y2-y3) - y1*(x2-x3) + (x2*y3 - x3*y2))
    
    def det3(xa, ya, xb, yb, xc, yc):
        return xa*(yb - yc) - ya*(xb - xc) + (xb*yc - xc*yb)
    
    det = (A * det3(x2, y2, x3, y3, x4, y4)
           - B * det3(x1, y1, x3, y3, x4, y4)
           + C * det3(x1, y1, x2, y2, x4, y4)
           - E * det3(x1, y1, x2, y2, x3, y3))
    
    return det == 0

def _validate(points, D):
    """Check all conditions for a valid set of points."""
    n = len(points)
    if n < 3:
        return False
    
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False
    
    # Check rational distances
    for i in range(n):
        for j in range(i+1, n):
            if not _is_rational_dist(points[i], points[j], D):
                return False
    
    # Check no three collinear
    for i, j, k in combinations(range(n), 3):
        if _collinear(points[i], points[j], points[k]):
            return False
    
    # Check no four concyclic
    for i, j, k, l in combinations(range(n), 4):
        if _concyclic(points[i], points[j], points[k], points[l], D):
            return False
    
    return True

def _hilbert_parameterization(D):
    """Generate points on a circle in Q(sqrt(D)) using Hilbert's 90 theorem approach.
    
    Points at rational distance from origin: x^2 + D*y^2 = r^2 for rational r.
    This is a conic. We can parameterize it.
    
    For D=1, this is x^2 + y^2 = r^2, parameterized by:
    x = r*(1-t^2)/(1+t^2), y = r*2t/(1+t^2)
    
    For general D, x^2 + D*y^2 = r^2.
    Let y' = sqrt(D)*y, then x^2 + y'^2 = r^2.
    But we need y rational, so y' = sqrt(D)*y.
    
    Parameterization:
    x = r*(1 - D*t^2)/(1 + D*t^2)
    y = r*2t/(1 + D*t^2)
    
    where t is rational.
    """
    points = []
    # Try different radii and parameter values
    for r in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for t_num in range(-10, 11):
            for t_den in range(1, 11):
                if gcd(t_num, t_den) != 1:
                    continue
                t = Fraction(t_num, t_den)
                denom = 1 + D * t * t
                if denom == 0:
                    continue
                x = r * (1 - D * t * t) / denom
                y = r * 2 * t / denom
                
                # Check if x and y are rational (they should be by construction)
                # and within bounds
                if abs(x.numerator) > 10**40 or abs(x.denominator) > 10**40:
                    continue
                if abs(y.numerator) > 10**40 or abs(y.denominator) > 10**40:
                    continue
                
                points.append((x, y))
    
    return points

def _search_for_set(D, max_points=12, time_limit=100):
    """Search for a valid set of points."""
    start_time = time.time()
    
    # Generate candidate points
    candidates = _hilbert_parameterization(D)
    
    # Also add some integer points for D=1
    if D == 1:
        for x in range(-20, 21):
            for y in range(-20, 21):
                candidates.append((Fraction(x), Fraction(y)))
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Try to build a valid set greedily
    best_set = []
    
    # Try different orderings
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Shuffle candidates
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        for p in shuffled:
            if time.time() - start_time > time_limit:
                break
            
            # Check if adding p maintains validity
            test_set = current_set + [p]
            
            # Quick checks
            valid = True
            for q in current_set:
                if p == q:
                    valid = False
                    break
                if not _is_rational_dist(p, q, D):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check collinearity with pairs
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    if _collinear(p, current_set[i], current_set[j]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check concyclicity with triples
            if len(current_set) >= 3:
                for i, j, k in combinations(range(len(current_set)), 3):
                    if _concyclic(p, current_set[i], current_set[j], current_set[k], D):
                        valid = False
                        break
            if not valid:
                continue
            
            current_set.append(p)
            
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            
            if len(best_set) >= max_points:
                break
        
        if len(best_set) >= max_points:
            break
    
    return best_set

def construct():
    """Construct a set of integral point sets in general position."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try D=1 first (integer points)
    best_result = None
    best_count = 0
    
    # Known 5-point set
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if _validate([(Fraction(x), Fraction(y)) for x, y in points_5], 1):
        best_result = points_5
        best_count = 5
    
    # Try to find larger sets
    for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97]:
        if time.time() > deadline - 5:
            break
        
        if not _is_squarefree(D):
            continue
        
        # Search for a set with this D
        found = _search_for_set(D, max_points=12, time_limit=min(30, deadline - time.time() - 5))
        
        if found and len(found) > best_count:
            best_count = len(found)
            best_result = found
        
        if best_count >= 8:
            break
    
    # If we didn't find anything better, use the known 5-point set
    if best_result is None:
        best_result = points_5
    
    # Format the result
    points_formatted = []
    for x, y in best_result:
        if isinstance(x, Fraction):
            x_num, x_den = x.numerator, x.denominator
        else:
            x_num, x_den = int(x), 1
        if isinstance(y, Fraction):
            y_num, y_den = y.numerator, y.denominator
        else:
            y_num, y_den = int(y), 1
        points_formatted.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': 1 if best_result is points_5 else D, 'points': points_formatted}

# EVOLVE-BLOCK-END
