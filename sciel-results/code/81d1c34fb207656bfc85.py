"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import isqrt, gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_square(n):
    """Check if n is a perfect square (n >= 0)."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist2_rational(x1, y1, x2, y2, D):
    """Return the squared distance as a Fraction, or None if not rational-squared.
    Points are (x, y*sqrt(D)) where x,y are Fractions.
    dist^2 = (x1-x2)^2 + D*(y1-y2)^2
    We need this to be a perfect square of a rational."""
    dx = x1 - x2
    dy = y1 - y2
    d2 = dx * dx + D * dy * dy
    # Check if d2 is a perfect square of a rational
    # d2 is a Fraction. Let d2 = p/q in lowest terms.
    # d2 = (a/b)^2 means p/q = a^2/b^2, so p must be a perfect square and q must be a perfect square.
    p = d2.numerator
    q = d2.denominator
    if _is_square(p) and _is_square(q):
        return d2
    return None

def _check_all_rational_distances(points, D):
    """Check all pairwise distances are rational. Points are list of (Fraction, Fraction)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if _dist2_rational(points[i][0], points[i][1], points[j][0], points[j][1], D) is None:
                return False
    return True

def _are_collinear(p1, p2, p3):
    """Check if three points are collinear. Points are (Fraction, Fraction)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # Cross product: (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

def _check_no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic. Points are (Fraction, Fraction).
    Four points are concyclic iff the determinant of the circle equation is zero.
    For points (xi, yi), the circle equation is x^2 + y^2 + Dx + Ey + F = 0.
    Four points are concyclic iff the 4x4 determinant is zero."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Check if points i,j,k,l are concyclic
                    # Using the determinant method
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # The four points are concyclic iff the following determinant is zero:
                    # | x1^2+y1^2  x1  y1  1 |
                    # | x2^2+y2^2  x2  y2  1 |
                    # | x3^2+y3^2  x3  y3  1 | = 0
                    # | x4^2+y4^2  x4  y4  1 |
                    # We compute this using Cramer's rule / direct expansion
                    # For efficiency, we can check if point 4 lies on the circle through 1,2,3
                    # Circle through (x1,y1), (x2,y2), (x3,y3):
                    # Solve for D, E, F in: x^2+y^2 + Dx + Ey + F = 0
                    # (x1^2+y1^2) + D*x1 + E*y1 + F = 0
                    # (x2^2+y2^2) + D*x2 + E*y2 + F = 0
                    # (x3^2+y3^2) + D*x3 + E*y3 + F = 0
                    # Subtract first from second and third:
                    # D*(x2-x1) + E*(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
                    # D*(x3-x1) + E*(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
                    a1 = x2 - x1
                    b1 = y2 - y1
                    c1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
                    a2 = x3 - x1
                    b2 = y3 - y1
                    c2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
                    det = a1 * b2 - a2 * b1
                    if det == 0:
                        # Points 1,2,3 are collinear, already handled
                        continue
                    D_val = (c1 * b2 - c2 * b1) / det
                    E_val = (a1 * c2 - a2 * c1) / det
                    F_val = -(x1*x1 + y1*y1) - D_val * x1 - E_val * y1
                    # Check if point 4 satisfies the circle equation
                    lhs = x4*x4 + y4*y4 + D_val*x4 + E_val*y4 + F_val
                    if lhs == 0:
                        return False
    return True

def _verify(points, D):
    """Full verification of all constraints."""
    if not _check_all_rational_distances(points, D):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points):
        return False
    return True

def _search_6th_point(base_points, D, time_limit, start_time):
    """Search for a 6th point to add to base_points."""
    best = None
    # Try integer points in a range
    # The base points span roughly x in [-56, 16], y in [0, 63]
    # Try a wider range
    for x in range(-200, 201):
        for y in range(-200, 201):
            if time.time() - start_time > time_limit:
                break
            new_pt = (Fraction(x), Fraction(y))
            # Quick check: distance to each base point
            valid = True
            for bp in base_points:
                if _dist2_rational(bp[0], bp[1], new_pt[0], new_pt[1], D) is None:
                    valid = False
                    break
            if not valid:
                continue
            # Check no three collinear
            test_pts = base_points + [new_pt]
            if not _check_no_three_collinear(test_pts):
                continue
            # Check no four concyclic
            if not _check_no_four_concyclic(test_pts):
                continue
            # Full verification
            if _verify(test_pts, D):
                return test_pts
    return None

def _search_rational_6th_point(base_points, D, time_limit, start_time):
    """Search for a 6th point with rational coordinates (not just integer)."""
    # Try points with small denominators
    for den in range(1, 20):
        for x_num in range(-400, 401, 1):
            for y_num in range(-400, 401, 1):
                if time.time() - start_time > time_limit:
                    return None
                # Reduce fraction
                x = Fraction(x_num, den)
                y = Fraction(y_num, den)
                new_pt = (x, y)
                valid = True
                for bp in base_points:
                    if _dist2_rational(bp[0], bp[1], new_pt[0], new_pt[1], D) is None:
                        valid = False
                        break
                if not valid:
                    continue
                test_pts = base_points + [new_pt]
                if not _check_no_three_collinear(test_pts):
                    continue
                if not _check_no_four_concyclic(test_pts):
                    continue
                if _verify(test_pts, D):
                    return test_pts
    return None

def construct():
    start_time = time.time()
    total_time_limit = 120  # seconds, with safety margin
    
    # Start with the known 5-point set
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    D = 1
    
    # Verify the base set
    assert _verify(base_points, D), "Base set should be valid"
    
    best_points = base_points
    best_D = D
    
    # Try to find a 6th point with D=1
    time_remaining = total_time_limit - (time.time() - start_time)
    if time_remaining > 5:
        result = _search_6th_point(base_points, D, time_remaining - 5, start_time)
        if result is not None:
            best_points = result
            best_D = D
    
    # If still 5 points, try searching with rational coordinates
    if len(best_points) == 5:
        time_remaining = total_time_limit - (time.time() - start_time)
        if time_remaining > 5:
            result = _search_rational_6th_point(base_points, D, time_remaining - 5, start_time)
            if result is not None:
                best_points = result
                best_D = D
    
    # Format the result
    points_out = [[[int(x.numerator), int(x.denominator)], [int(y.numerator), int(y.denominator)]] 
                  for x, y in best_points]
    
    return {'D': best_D, 'points': points_out}
# EVOLVE-BLOCK-END
