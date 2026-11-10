"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import math
import time
import random

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _check_collinear(p1, p2, p3, D):
    # Points are (x1, y1*sqrt(D)), etc.
    # Collinear iff determinant is zero:
    # |x1 y1*sqrt(D) 1|
    # |x2 y2*sqrt(D) 1| = 0
    # |x3 y3*sqrt(D) 1|
    # Expanding: x1*(y2-y3)*sqrt(D) - y1*sqrt(D)*(x2-x3) + (x2*y3 - x3*y2)*sqrt(D) = 0
    # Divide by sqrt(D) (assuming D != 0):
    # x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2 = 0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    val = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    return val == 0

def _check_concyclic(p1, p2, p3, p4, D):
    # Four points are concyclic iff the determinant of the 4x4 matrix is zero.
    # For points (xi, yi*sqrt(D)), the circle equation is:
    # x^2 + y^2*D + Ax + By*sqrt(D) + C = 0
    # We can eliminate sqrt(D) by considering the system.
    # Actually, let's use the standard concyclic test adapted for this metric.
    # Distance squared between (x1, y1*sqrt(D)) and (x2, y2*sqrt(D)) is:
    # d^2 = (x1-x2)^2 + D*(y1-y2)^2
    # Points are concyclic iff the Cayley-Menger determinant is 0, or simpler:
    # There exist A, B, C such that for all i:
    # xi^2 + yi^2*D + A*xi + B*yi*sqrt(D) + C = 0
    # This is a linear system in A, B*sqrt(D), C for each point.
    # Let's set up the system for p1, p2, p3 to find A, B, C (with B*sqrt(D) as a variable).
    # Let Z = B*sqrt(D). Then:
    # xi^2 + yi^2*D + A*xi + Z*yi + C = 0
    # This is linear in A, Z, C.
    # For p1, p2, p3:
    # x1^2 + y1^2*D + A*x1 + Z*y1 + C = 0
    # x2^2 + y2^2*D + A*x2 + Z*y2 + C = 0
    # x3^2 + y3^2*D + A*x3 + Z*y3 + C = 0
    # Subtract p1 from p2 and p3:
    # (x2^2-x1^2) + D*(y2^2-y1^2) + A*(x2-x1) + Z*(y2-y1) = 0
    # (x3^2-x1^2) + D*(y3^2-y1^2) + A*(x3-x1) + Z*(y3-y1) = 0
    # This is a 2x2 system for A and Z.
    # If the determinant is non-zero, we can solve for A and Z.
    # Then check if p4 satisfies the equation.
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Coefficients for A and Z
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = (x2*x2 - x1*x1) + D * (y2*y2 - y1*y1)
    
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = (x3*x3 - x1*x1) + D * (y3*y3 - y1*y1)
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        # Points 1, 2, 3 are collinear (or coincident), which shouldn't happen in valid sets
        # But if they are, they don't define a unique circle.
        # If p4 is also collinear, they are "concyclic" in a degenerate sense?
        # The problem says "no four concyclic". Usually this implies distinct circles.
        # If 3 are collinear, any line is a circle of infinite radius?
        # Let's assume general position implies no 3 collinear, so det != 0.
        # If det == 0, we can't determine the circle uniquely from these 3.
        # But if the input is valid, this shouldn't happen.
        # For safety, if det==0, we check if p4 is collinear with them too.
        return _check_collinear(p1, p2, p4, D) or _check_collinear(p1, p3, p4, D) or _check_collinear(p2, p3, p4, D)
    
    # Solve for A and Z
    # A * a1 + Z * b1 = -c1
    # A * a2 + Z * b2 = -c2
    # A = (-c1*b2 + c2*b1) / det
    # Z = (c1*a2 - c2*a1) / det
    
    # We want to check if p4 satisfies:
    # x4^2 + y4^2*D + A*x4 + Z*y4 + C = 0
    # We need C. From p1:
    # C = -(x1^2 + y1^2*D + A*x1 + Z*y1)
    # So the condition is:
    # x4^2 + y4^2*D + A*x4 + Z*y4 - (x1^2 + y1^2*D + A*x1 + Z*y1) = 0
    # (x4^2 - x1^2) + D*(y4^2 - y1^2) + A*(x4 - x1) + Z*(y4 - y1) = 0
    
    # Substitute A and Z:
    # A = (-c1*b2 + c2*b1) / det
    # Z = (c1*a2 - c2*a1) / det
    
    # Let's compute the numerator of the LHS for p4:
    # N = det * [ (x4^2 - x1^2) + D*(y4^2 - y1^2) ] + (-c1*b2 + c2*b1)*(x4 - x1) + (c1*a2 - c2*a1)*(y4 - y1)
    
    term0 = (x4*x4 - x1*x1) + D * (y4*y4 - y1*y1)
    term1 = (-c1*b2 + c2*b1) * (x4 - x1)
    term2 = (c1*a2 - c2*a1) * (y4 - y1)
    
    N = det * term0 + term1 + term2
    
    return N == 0

def _dist_sq(p1, p2, D):
    x1, y1 = p1
    x2, y2 = p2
    return (x1 - x2) * (x1 - x2) + D * (y1 - y2) * (y1 - y2)

def _valid_add(points, new_point, D):
    # Check if adding new_point to points maintains general position
    # and rational distances.
    
    # 1. Check rational distance to all existing points
    for p in points:
        d2 = _dist_sq(p, new_point, D)
        if not _is_square(d2):
            return False
    
    # 2. Check no three collinear
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _check_collinear(points[i], points[j], new_point, D):
                return False
    
    # 3. Check no four concyclic
    if len(points) >= 3:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    if _check_concyclic(points[i], points[j], points[k], new_point, D):
                        return False
    
    return True

def _try_build_lattice(D, max_time=110):
    # Try to build a set of lattice points (integer x, y) with rational distances.
    # This is equivalent to finding points with integer distances.
    # We use a greedy backtracking search.
    
    best_set = []
    
    # Generate candidate points.
    # We can search in a box.
    # Let's try a few random starting points and then expand.
    
    # A known 5-point set:
    # (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    # Let's verify this one first.
    known = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if all(_valid_add(known[:i], known[i], D) for i in range(1, len(known))):
        best_set = known[:]
    
    # Now try to extend.
    # We can search for a 6th point.
    # Let's try a local search around the existing points.
    
    # Define a set of candidate points to try.
    # We can generate points in a grid and filter by distance constraints.
    
    # To be efficient, we can precompute which points are at rational distance from all points in the current set.
    
    # Let's try a random restart strategy with a time limit.
    start_time = time.time()
    
    # Try to find a 6th point.
    # We can search in a larger grid.
    
    # For simplicity, let's just try to extend the known set with a few random points.
    # Or better, use a systematic search.
    
    # Let's try to find points that are at integer distance from (0,0).
    # These are solutions to x^2 + y^2 = k^2.
    # Pythagorean triples.
    
    # Generate some Pythagorean triples.
    pyth_points = set()
    for m in range(1, 50):
        for n in range(1, m):
            if (m - n) % 2 == 1 and _gcd(m, n) == 1:
                x = m * m - n * n
                y = 2 * m * n
                z = m * m + n * n
                pyth_points.add((x, y))
                pyth_points.add((x, -y))
                pyth_points.add((-x, y))
                pyth_points.add((-x, -y))
                pyth_points.add((y, x))
                pyth_points.add((y, -x))
                pyth_points.add((-y, x))
                pyth_points.add((-y, -x))
    
    # Filter these points to those that are at rational distance from all points in the best set.
    # And check general position.
    
    for p in pyth_points:
        if p in best_set:
            continue
        if _valid_add(best_set, p, D):
            best_set.append(p)
            if len(best_set) >= 6:
                break
        if time.time() - start_time > max_time - 10:
            break
    
    return best_set

def construct():
    D = 1
    # Try to build a set with D=1 (lattice points).
    points = _try_build_lattice(D)
    
    # Format the output
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
