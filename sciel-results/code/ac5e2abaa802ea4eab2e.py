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

def _find_valid_pairs(D, max_coord=200):
    """Find integer pairs (dx, dy) such that dx^2 + D*dy^2 is a perfect square."""
    pairs = set()
    for dx in range(-max_coord, max_coord + 1):
        for dy in range(-max_coord, max_coord + 1):
            if dx == 0 and dy == 0:
                continue
            val = dx * dx + D * dy * dy
            if _is_perfect_square(val):
                pairs.add((dx, dy))
    return pairs

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
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(6), Fraction(8))]
        # That has collinear points, use different
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        best_D = 1
        best_count = 3

    random.seed(12345)

    # Strategy: For each candidate D, try to find 6+ point sets
    # Focus on D values where the Diophantine equation is rich in solutions
    candidate_Ds = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97]
    
    for D in candidate_Ds:
        if time.time() > deadline:
            break
        if not _is_squarefree(D):
            continue
        
        # For D=2, the equation a^2 + 2b^2 = c^2 has many solutions
        # Try systematic search for 6 points
        max_coord = 100
        attempts = 0
        max_attempts = 20000
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Generate random points
            pts = []
            for _ in range(6):
                x = Fraction(random.randint(-max_coord, max_coord), 1)
                y = Fraction(random.randint(-max_coord, max_coord), 1)
                pts.append((x, y))
            
            if _validate(pts, D):
                best_points = pts
                best_D = D
                best_count = 6
                # Try to extend to 7
                if time.time() < deadline:
                    extended = False
                    for _ in range(5000):
                        if time.time() > deadline:
                            break
                        x = Fraction(random.randint(-max_coord * 2, max_coord * 2), 1)
                        y = Fraction(random.randint(-max_coord * 2, max_coord * 2), 1)
                        new_pts = pts + [(x, y)]
                        if _validate(new_pts, D):
                            best_points = new_pts
                            best_count = 7
                            extended = True
                            break
                    if extended and time.time() < deadline:
                        # Try 8
                        for _ in range(2000):
                            if time.time() > deadline:
                                break
                            x = Fraction(random.randint(-max_coord * 3, max_coord * 3), 1)
                            y = Fraction(random.randint(-max_coord * 3, max_coord * 3), 1)
                            new_pts = best_points + [(x, y)]
                            if _validate(new_pts, D):
                                best_points = new_pts
                                best_count = 8
                                break
                break
    
    # Also try D=1 with larger search space
    if time.time() < deadline and best_count < 6:
        max_coord = 500
        for _ in range(50000):
            if time.time() > deadline:
                break
            pts = []
            for _ in range(6):
                x = Fraction(random.randint(-max_coord, max_coord), 1)
                y = Fraction(random.randint(-max_coord, max_coord), 1)
                pts.append((x, y))
            if _validate(pts, 1):
                best_points = pts
                best_D = 1
                best_count = 6
                break

    # Try adding a 6th point to the known 5-point set for D=1
    if time.time() < deadline and best_count == 5:
        base = known_points
        for _ in range(100000):
            if time.time() > deadline:
                break
            x = Fraction(random.randint(-200, 200), 1)
            y = Fraction(random.randint(-200, 200), 1)
            new_pts = base + [(x, y)]
            if _validate(new_pts, 1):
                best_points = new_pts
                best_count = 6
                break

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
