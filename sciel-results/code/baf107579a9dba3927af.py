"""
Search for integral point sets in general position (no 3 collinear, no 4 concyclic)
in the plane with points (x, y*sqrt(D)) for rational x, y.
"""
import time
import random
from math import gcd
from fractions import Fraction

DEADLINE = 110.0  # safety margin before 120s kill

def _is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    for c in (r - 1, r, r + 1):
        if c >= 0 and c * c == n:
            return True
    return False

def _sq_dist(p, q, D):
    """Squared distance between points p=(x1,y1), q=(x2,y2) in (x, y*sqrt(D)) space.
    Returns Fraction."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _dist_rational(p, q, D):
    """Check if squared distance is a perfect square (of a rational)."""
    d2 = _sq_dist(p, q, D)
    num = d2.numerator
    den = d2.denominator
    if _is_square(num) and _is_square(den):
        return True
    return False

def _no_three_collinear(points):
    """Check no three points are collinear using cross product = 0."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic.
    Four points (x1,y1),...,(x4,y4) are concyclic iff the determinant of
    the circle equation is zero. For points in (x, y*sqrt(D)) space, we work
    in transformed coordinates. A circle in this space has equation:
    x^2 + D*y^2 + Ax + By + C = 0
    Four points are concyclic iff the 4x4 determinant:
    | x1^2+D*y1^2  x1  y1  1 |
    | x2^2+D*y2^2  x2  y2  1 | = 0
    | x3^2+D*y3^2  x3  y3  1 |
    | x4^2+D*y4^2  x4  y4  1 |
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Build 4x4 matrix
                    M = []
                    for p in (p1, p2, p3, p4):
                        row = [p[0]*p[0] + D*p[1]*p[1], p[0], p[1], Fraction(1)]
                        M.append(row)
                    # Compute determinant via cofactor expansion (4x4)
                    # Using Sarrus-like for 4x4: det = sum over permutations
                    # Simpler: expand along last row
                    def det4(m):
                        # m is 4x4 list of Fractions
                        a = m[0]; b = m[1]; c = m[2]; d = m[3]
                        return (a[0]*(b[1]*c[2]*d[3] - b[1]*d[2]*c[3] - b[2]*c[1]*d[3] + b[2]*d[1]*c[3] + b[3]*c[1]*d[2] - b[3]*d[1]*c[2])
                              - a[1]*(b[0]*c[2]*d[3] - b[0]*d[2]*c[3] - b[2]*c[0]*d[3] + b[2]*d[0]*c[3] + b[3]*c[0]*d[2] - b[3]*d[0]*c[2])
                              + a[2]*(b[0]*c[1]*d[3] - b[0]*d[1]*c[3] - b[1]*c[0]*d[3] + b[1]*d[0]*c[3] + b[3]*c[0]*d[1] - b[3]*d[0]*c[1])
                              - a[3]*(b[0]*c[1]*d[2] - b[0]*d[1]*c[2] - b[1]*c[0]*d[2] + b[1]*d[0]*c[2] + b[2]*c[0]*d[1] - b[2]*d[0]*c[1]))
                    if det4(M) == 0:
                        return False
    return True

def _all_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_rational(points[i], points[j], D):
                return False
    return True

def _is_valid(points, D):
    if len(points) < 3:
        return False
    if not _all_distances_rational(points, D):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _search_lattice():
    """Search for integral point sets in Z^2 (D=1) using local search."""
    best = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_len = 5
    deadline = time.time() + DEADLINE - 10
    
    # Try to add points to the known 5-point set
    # Known 5-point set: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Try random lattice points and check if they can be added
    
    current = list(best)
    for attempt in range(200000):
        if time.time() > deadline:
            break
        # Random point in a reasonable range
        x = random.randint(-200, 200)
        y = random.randint(-200, 200)
        candidate = current + [(x, y)]
        if len(candidate) > 12:
            break
        if _is_valid(candidate, 1):
            current = candidate
            best = current[:]
            best_len = len(current)
            if best_len >= 7:
                break
    
    return best

def _search_with_D(D_val, max_coord=50):
    """Search for point sets with given D."""
    best = None
    deadline = time.time() + DEADLINE - 20
    
    # Start with simple configurations
    # Try small integer coordinates
    for attempt in range(50000):
        if time.time() > deadline:
            break
        n = random.randint(3, 6)
        points = []
        for _ in range(n):
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            points.append((Fraction(x), Fraction(y)))
        
        if _is_valid(points, D_val):
            if best is None or len(points) > len(best):
                best = points
                if len(points) >= 6:
                    break
    
    return best

def construct():
    start_time = time.time()
    
    # Start with the known 5-point integral set
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = known_points[:]
    best_D = 1
    best_len = 5
    
    # Verify the known set
    pts_frac = [(Fraction(x), Fraction(y)) for x, y in known_points]
    if _is_valid(pts_frac, 1):
        pass  # confirmed valid
    
    # Try to extend the known set
    current = list(pts_frac)
    deadline = start_time + DEADLINE - 5
    
    # Random search to add points
    for attempt in range(300000):
        if time.time() > deadline:
            break
        x = random.randint(-300, 300)
        y = random.randint(-300, 300)
        candidate = current + [(Fraction(x), Fraction(y))]
        if len(candidate) > 12:
            break
        if _is_valid(candidate, 1):
            current = candidate
            best_points = current[:]
            best_len = len(current)
            if best_len >= 8:
                break
    
    # Also try D=2, D=3 with smaller search
    if time.time() < deadline - 10:
        for D_val in [2, 3]:
            if time.time() > deadline - 5:
                break
            sub_deadline = min(deadline - 2, time.time() + 15)
            for attempt in range(30000):
                if time.time() > sub_deadline:
                    break
                n = random.randint(3, 5)
                points = []
                valid = True
                for _ in range(n):
                    x = random.randint(-30, 30)
                    y = random.randint(-30, 30)
                    points.append((Fraction(x), Fraction(y)))
                if _is_valid(points, D_val):
                    if len(points) > best_len:
                        best_points = points[:]
                        best_D = D_val
                        best_len = len(points)
                    break  # found one for this D, move on
    
    # Format result
    result_points = []
    for (x, y) in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}
