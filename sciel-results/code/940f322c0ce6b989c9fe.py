"""Search for integral point sets with rational distances, no 3 collinear, no 4 concyclic."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _check_pairwise_rational(points, D):
    """Check all pairwise distances are rational (exact)."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            # dist^2 = dx^2 + D*dy^2, must be a perfect square of a rational
            d2 = dx*dx + D*dy*dy
            # Check if d2 is a perfect square in Q
            num, den = d2.numerator, d2.denominator
            if num < 0 or den < 0:
                return False
            # sqrt(num/den) = sqrt(num*den)/den
            inner = num * den
            # Check inner is a perfect square
            if inner < 0:
                return False
            s = int(inner ** 0.5)
            if s*s != inner:
                # Try nearby
                s2 = s+1
                if s2*s2 != inner:
                    return False
    return True

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Cross product of (pj-pi) and (pk-pi)
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                cross = dx1*dy2 - dy1*dx2
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic. A circle in this geometry:
    x^2 + D*y^2 + Ax + By + C = 0 (generalized circle)
    Four points concyclic iff the 4x4 determinant vanishes."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    # Build 4x4 matrix: [x^2+D*y^2, x, y, 1] for each point
                    det = 0
                    # Compute determinant via expansion
                    rows = []
                    for idx in [i, j, k, m]:
                        x = points[idx][0]
                        y = points[idx][1]
                        rows.append([x*x + D*y*y, x, y, Fraction(1)])
                    # Determinant of 4x4
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

def _search_integer_points(deadline):
    """Search for integer points (D=1) with integer distances, no 3 collinear, no 4 concyclic."""
    best = 5
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend the known 5-point set by adding a 6th point
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Search for 6th point in a bounded region
    # For D=1, we need integer distances to all 5 existing points
    # This is a system of quadratic equations - search lattice points
    search_range = 200
    found_6 = None
    
    for x in range(-search_range, search_range+1):
        if time.time() > deadline:
            break
        for y in range(-search_range, search_range+1):
            if time.time() > deadline:
                break
            cand = (x, y)
            # Check distinct from existing
            if cand in base:
                continue
            # Check integer distances to all base points
            ok = True
            for px, py in base:
                dx = x - px
                dy = y - py
                d2 = dx*dx + dy*dy
                s = int(d2 ** 0.5)
                if s*s != d2:
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear with the 5 base + new point
            new_points = base + [cand]
            if not _no_three_collinear(new_points):
                continue
            # Check no four concyclic
            if not _no_four_concyclic(new_points, 1):
                continue
            found_6 = new_points
            break
        if found_6:
            break
    
    if found_6:
        return 6, found_6
    
    # Also try a broader random search for 6 points
    import random
    random.seed(42)
    for attempt in range(50000):
        if time.time() > deadline:
            break
        pts = []
        for _ in range(6):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pts.append((x, y))
        # Remove duplicates
        pts = list(set(pts))
        if len(pts) < 6:
            continue
        # Check integer distances
        ok = True
        for i in range(6):
            for j in range(i+1, 6):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx*dx + dy*dy
                s = int(d2 ** 0.5)
                if s*s != d2:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        if not _no_three_collinear(pts):
            continue
        if not _no_four_concyclic(pts, 1):
            continue
        return 6, pts
    
    return best, best_points

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Try to find 6+ points
    n_pts, pts = _search_integer_points(deadline)
    
    if time.time() > deadline - 5:
        # Fallback to known 5-point set
        pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        n_pts = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
