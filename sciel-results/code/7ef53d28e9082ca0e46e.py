"""Search for integral point sets with rational distances, no 3 collinear, no 4 concyclic."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def _is_perfect_square(n):
    """Check if n is a perfect square (exact, for non-negative integers)."""
    if n < 0:
        return False
    if n == 0:
        return True
    s = int(math.isqrt(n))
    return s * s == n

def _check_pairwise_rational(points, D):
    """Check all pairwise distances are rational (exact)."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + D*dy*dy
            num, den = d2.numerator, d2.denominator
            if num < 0 or den < 0:
                return False
            inner = num * den
            if not _is_perfect_square(inner):
                return False
    return True

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                cross = dx1*dy2 - dy1*dx2
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    rows = []
                    for idx in [i, j, k, m]:
                        x = points[idx][0]
                        y = points[idx][1]
                        rows.append([x*x + D*y*y, x, y, Fraction(1)])
                    a = rows[0]; b = rows[1]; c = rows[2]; d = rows[3]
                    det = (a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[1]*d[3]-c[3]*d[1]) + b[3]*(c[1]*d[2]-c[2]*d[1]))
                         - a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[2]-c[2]*d[0]))
                         + a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1]) - b[1]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[1]-c[1]*d[0]))
                         - a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1]) - b[1]*(c[0]*d[2]-c[2]*d[0]) + b[2]*(c[0]*d[1]-c[1]*d[0])))
                    if det == 0:
                        return False
    return True

def _valid_set(points, D):
    """Full validation."""
    if not _check_pairwise_rational(points, D):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _try_extend_base(base, D, deadline):
    """Try to add points to base set, searching for valid extensions."""
    current = list(base)
    for target_n in range(len(current)+1, 13):
        if time.time() > deadline:
            break
        found = False
        # Search for a point that can be added
        search_range = 300 if target_n <= 7 else 500
        for x in range(-search_range, search_range+1):
            if time.time() > deadline:
                break
            for y in range(-search_range, search_range+1):
                if time.time() > deadline:
                    break
                cand = (Fraction(x), Fraction(y))
                if cand in current:
                    continue
                # Check distances to all current points
                ok = True
                for px, py in current:
                    dx = cand[0] - px
                    dy = cand[1] - py
                    d2 = dx*dx + D*dy*dy
                    num, den = d2.numerator, d2.denominator
                    inner = num * den
                    if not _is_perfect_square(inner):
                        ok = False
                        break
                if not ok:
                    continue
                # Check no three collinear
                new_pts = current + [cand]
                if not _no_three_collinear(new_pts):
                    continue
                # Check no four concyclic
                if not _no_four_concyclic(new_pts, D):
                    continue
                current = new_pts
                found = True
                break
            if found:
                break
        if not found:
            break
    return current

def _search_integer_points(deadline):
    """Search for integer points (D=1) with integer distances."""
    # Known 5-point set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend
    extended = _try_extend_base(base5, 1, deadline)
    if len(extended) >= 6:
        return len(extended), extended
    
    # Try different base configurations
    # Known 4-point sets that might extend better
    bases = [
        [(0, 0), (3, 4), (0, 8), (3, 12)],
        [(0, 0), (5, 0), (0, 5), (5, 5)],
        [(0, 0), (9, 0), (0, 12), (9, 12)],
        [(0, 0), (7, 24), (0, 25), (7, 49)],
    ]
    
    for base in bases:
        if time.time() > deadline:
            break
        ext = _try_extend_base(base, 1, deadline)
        if len(ext) > len(extended):
            extended = ext
        if len(extended) >= 6:
            break
    
    return len(extended), extended

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin
    
    # Try to find 6+ points with D=1
    n_pts, pts = _search_integer_points(deadline)
    
    # Fallback to known 5-point set if needed
    if n_pts < 5 or time.time() > deadline - 2:
        pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        n_pts = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
