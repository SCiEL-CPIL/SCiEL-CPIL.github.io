"""Rewritten program for integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_DEADLINE = 150.0
_START_TIME = None

def _now():
    return time.time() - _START_TIME

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sqfree(n):
    # return squarefree part of n (n>=0)
    if n == 0:
        return 0
    res = 1
    p = 2
    temp = n
    while p * p <= temp:
        if temp % (p * p) == 0:
            while temp % (p * p) == 0:
                temp //= (p * p)
            res *= p
        elif temp % p == 0:
            temp //= p
        p += 1 if p == 2 else 2
    if temp > 1:
        res *= temp
    return res

def _dist_sq(p1, p2, D):
    # p1, p2 are (x, y) with y*sqrt(D)
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy * D

def _is_rational_dist_sq(v, D):
    # v is an integer (or rational represented as Fraction) representing dx^2 + dy^2 * D
    # We need v to be a rational square in Q. Since v is integer here (if D integer and coords integer),
    # it must be a perfect square.
    # But wait, coords are rational. Let's keep coords as Fractions.
    # dist^2 = (x1-x2)^2 + D*(y1-y2)^2.
    # If x, y are Fractions, dist^2 is a Fraction.
    # We need dist^2 to be a square of a rational number.
    # Let dist^2 = a/b (reduced). Then a and b must be perfect squares.
    if v < 0:
        return False
    num = v.numerator
    den = v.denominator
    return _is_square(num) and _is_square(den)

def _check_no_three_collinear(points):
    # points: list of (x, y) Fractions
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    # A circle in this geometry: x^2 + y^2 * D + A x + B y * D + C = 0
    # (General conic for circle in metric with y scaled by sqrt(D))
    # Actually, the equation of a circle passing through (x1,y1), (x2,y2), (x3,y3) is:
    # (x^2 + D y^2 + A x + B D y + C) = 0 ?
    # Let's derive. Distance squared from (x,y) to (x0,y0) is (x-x0)^2 + D(y-y0)^2.
    # Circle center (x0,y0), radius r: (x-x0)^2 + D(y-y0)^2 = r^2
    # x^2 - 2x0 x + x0^2 + D y^2 - 2 D y0 y + D y0^2 - r^2 = 0
    # x^2 + D y^2 - 2x0 x - 2 D y0 y + (x0^2 + D y0^2 - r^2) = 0
    # So the equation is x^2 + D y^2 + A x + B y + C = 0 where A=-2x0, B=-2D y0.
    # Note: The coefficient of y is B, not B*D. Let's be precise.
    # Term is -2 D y0 y. So coefficient of y is -2 D y0.
    # Let's use the determinant form for 4 points to be concyclic.
    # 4 points (xi, yi) are concyclic iff:
    # | x1^2 + D y1^2, x1, D y1, 1 |
    # | x2^2 + D y2^2, x2, D y2, 1 | = 0
    # | x3^2 + D y3^2, x3, D y3, 1 |
    # | x4^2 + D y4^2, x4, D y4, 1 |
    # Wait, is the column for y "D*y" or "y"?
    # Equation: x^2 + D y^2 + A x + B (D y) + C = 0.
    # Let U = D y. Then x^2 + U^2 + A x + B U + C = 0.
    # This is a standard circle in (x, U) coordinates.
    # So yes, the column should be D*y.
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Matrix rows
                    r1 = [p1[0]*p1[0] + D*p1[1]*p1[1], p1[0], D*p1[1], 1]
                    r2 = [p2[0]*p2[0] + D*p2[1]*p2[1], p2[0], D*p2[1], 1]
                    r3 = [p3[0]*p3[0] + D*p3[1]*p3[1], p3[0], D*p3[1], 1]
                    r4 = [p4[0]*p4[0] + D*p4[1]*p4[1], p4[0], D*p4[1], 1]
                    # Determinant
                    # Use Sarrus rule or expansion for 4x4.
                    # det = sum of permutations ...
                    # Let's implement a small 4x4 determinant calculator using Fractions.
                    def det4(m):
                        return (
                            m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                            - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                            + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                            - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                        )
                    m = [r1, r2, r3, r4]
                    if det4(m) == 0:
                        return False
    return True

def _validate(points, D):
    # points: list of (x, y) Fractions
    # Check distinct
    set_pts = set(points)
    if len(set_pts) != len(points):
        return False
    # Check distances
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d2 = _dist_sq(points[i], points[j], D)
            if not _is_rational_dist_sq(d2, D):
                return False
    # Check general position
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points = [ (Fraction(x), Fraction(y)) for x, y in base_points ]
    
    best_points = points
    best_D = D
    
    # Try to add a 6th point
    # Search space: x, y in some range
    # To keep it fast, we can search over a grid or random points.
    # Since D=1, we are looking for integer points (or rational) such that distances are rational.
    # For integer points, distances are rational iff squared distances are perfect squares.
    
    # Let's try a local search from the base points.
    # We can try to add a point (x, y) where x, y are integers in [-200, 200].
    
    def try_add_point(pts, D, x, y):
        new_pt = (Fraction(x), Fraction(y))
        if new_pt in pts:
            return None
        new_pts = pts + [new_pt]
        if _validate(new_pts, D):
            return new_pts
        return None

    # Search for 6th point
    found_6 = False
    # Limit search time
    max_time = 140.0
    
    # We can iterate over candidate points.
    # To be efficient, pre-filter candidates that have rational distance to all existing points.
    # For D=1 and integer coords, dist^2 = (x1-x2)^2 + (y1-y2)^2.
    # This must be a perfect square.
    
    candidates = []
    # Generate candidates in a box
    # Let's use a box around the existing points.
    # Existing x range: -56 to 16. y range: 0 to 63.
    # Let's search x in [-100, 100], y in [-50, 150].
    
    # To avoid O(N^2) checks for every candidate, we can check distances to base points first.
    base_ints = [(int(x), int(y)) for x, y in base_points]
    
    for x in range(-100, 101):
        if _now() > max_time:
            break
        for y in range(-50, 151):
            if _now() > max_time:
                break
            pt = (x, y)
            if pt in base_ints:
                continue
            ok = True
            for bx, by in base_ints:
                d2 = (x - bx)**2 + (y - by)**2
                if not _is_square(d2):
                    ok = False
                    break
            if ok:
                candidates.append(pt)
    
    # Now check general position for each candidate
    for cx, cy in candidates:
        if _now() > max_time:
            break
        new_pts = [ (Fraction(x), Fraction(y)) for x, y in base_points ] + [ (Fraction(cx), Fraction(cy)) ]
        if _validate(new_pts, D):
            best_points = new_pts
            found_6 = True
            # Try to add a 7th point?
            # This would be much harder.
            # Let's stick with 6 if found.
            break
    
    # If we found 6, we might try to find 7, but that's very expensive.
    # For now, return the best found.
    
    # Convert back to required format
    # points: [[[x_num, x_den], [y_num, y_den]], ...]
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
