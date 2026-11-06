"""Integral point sets in general position via circle parametrization + rational transform."""
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

def _is_squarefree(n):
    n = abs(n)
    if n == 0:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _circle_point(t_num, t_den):
    """Rational parametrization of unit circle: ((1-t^2)/(1+t^2), 2t/(1+t^2))."""
    t = Fraction(t_num, t_den)
    t2 = t * t
    denom = 1 + t2
    x = (1 - t2) / denom
    y = 2 * t / denom
    return (x, y)

def _rational_distance_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_perfect_square_frac(f):
    """Check if Fraction f is a perfect square of a rational."""
    if f < 0:
        return False
    num = f.numerator
    den = f.denominator
    g = gcd(num, den)
    num_s = num // g
    den_s = den // g
    sqrt_num = isqrt(num_s)
    sqrt_den = isqrt(den_s)
    return sqrt_num * sqrt_num == num_s and sqrt_den * sqrt_den == den_s

def _check_general_position(points):
    """Check no three collinear and no four concyclic using exact arithmetic."""
    n = len(points)
    
    # Check no three collinear
    for i, j, k in combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            return False
    
    # Check no four concyclic
    for i, j, k, l in combinations(range(n), 4):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        
        def det4(m):
            return (m[0][0]*det3([m[1][1:], m[2][1:], m[3][1:]])
                   - m[0][1]*det3([m[1][0:1]+m[1][2:], m[2][0:1]+m[2][2:], m[3][0:1]+m[3][2:]])
                   + m[0][2]*det3([m[1][0:2], m[2][0:2], m[3][0:2]])
                   - m[0][3]*det3([m[1][0:3], m[2][0:3], m[3][0:3]]))
        
        m = [
            [x1*x1 + y1*y1, x1, y1, 1],
            [x2*x2 + y2*y2, x2, y2, 1],
            [x3*x3 + y3*y3, x3, y3, 1],
            [x4*x4 + y4*y4, x4, y4, 1]
        ]
        if det4(m) == 0:
            return False
    
    return True

def _all_distances_rational(points):
    """Check all pairwise distances are rational."""
    for i, j in combinations(range(len(points)), 2):
        d2 = _rational_distance_sq(points[i], points[j])
        if not _is_perfect_square_frac(d2):
            return False
    return True

def _verify(points):
    """Full verification: distinct, rational distances, general position."""
    if len(points) < 3:
        return False
    # Check distinctness
    seen = set()
    for p in points:
        key = (p[0].numerator, p[0].denominator, p[1].numerator, p[1].denominator)
        if key in seen:
            return False
        seen.add(key)
    # Check rational distances
    if not _all_distances_rational(points):
        return False
    # Check general position
    if not _check_general_position(points):
        return False
    return True

def _apply_rational_transform(points, p, q, r, s):
    """Apply (x,y) -> ((px+qy+r)/(sx+ty+1), (ux+vy+w)/(sx+ty+1)) preserving rationality.
    We use a specific form that maps circles to non-circular conics."""
    # Use: (x', y') = ((x + a*y + b)/(1 + c*y), (y + d*x + e)/(1 + c*y))
    # This is a rational map that preserves the property that if original points
    # have rational coords, so do transformed points.
    # Key: choose parameters so that squared distances remain rational.
    result = []
    for x, y in points:
        denom = 1 + r * y
        if denom == 0:
            return None
        x_new = (p * x + q * y) / denom
        y_new = (s * x + y) / denom
        result.append((x_new, y_new))
    return result

def _find_best_set():
    """Search for a large set of points with rational distances in general position."""
    best_points = None
    best_size = 0
    
    # Generate candidate points on unit circle
    t_values = []
    for num in range(-10, 11):
        for den in range(1, 11):
            if gcd(abs(num), den) == 1:
                t_values.append(Fraction(num, den))
    
    # Also add some larger values
    for num in [-15, -12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 15]:
        t_values.append(Fraction(num, 1))
    
    circle_points = []
    for t in t_values:
        cp = _circle_point(t.numerator, t.denominator)
        circle_points.append(cp)
    
    # Deduplicate circle points
    unique_circle = []
    seen = set()
    for cp in circle_points:
        key = (cp[0].numerator, cp[0].denominator, cp[1].numerator, cp[1].denominator)
        if key not in seen:
            seen.add(key)
            unique_circle.append(cp)
    
    circle_points = unique_circle
    
    # Try various rational transformations
    transform_params = [
        (1, 0, 0, 0),   # identity
        (1, 1, 1, 0),
        (1, -1, 1, 0),
        (1, 0, 1, 1),
        (1, 0, -1, 1),
        (2, 1, 1, 0),
        (1, 2, 1, 0),
        (1, 1, 2, 1),
        (1, -1, 2, 1),
        (2, -1, 1, 2),
        (1, 2, -1, 2),
        (3, 1, 1, 1),
        (1, 3, 1, 1),
        (1, 1, 1, 1),
        (1, 1, -1, -1),
    ]
    
    for (p, q, r, s) in transform_params:
        transformed = _apply_rational_transform(circle_points, p, q, r, s)
        if transformed is None:
            continue
        
        # Greedily build largest valid subset
        # Sort by some heuristic and greedily add
        from itertools import combinations as comb
        n = len(transformed)
        
        # Try greedy: sort by angle or x-coordinate, then try to add points
        # First, filter to those with reasonable coordinate sizes
        filtered = []
        for pt in transformed:
            if (abs(pt[0].numerator) <= 10**20 and abs(pt[0].denominator) <= 10**20 and
                abs(pt[1].numerator) <= 10**20 and abs(pt[1].denominator) <= 10**20):
                filtered.append(pt)
        
        # Try different orderings
        orderings = [
            lambda pts: sorted(pts, key=lambda p: (p[0].numerator, p[0].denominator)),
            lambda pts: sorted(pts, key=lambda p: (p[1].numerator, p[1].denominator)),
            lambda pts: sorted(pts, key=lambda p: (p[0] + p[1])),
        ]
        
        for order_fn in orderings:
            ordered = order_fn(filtered)
            current_set = []
            for pt in ordered:
                candidate = current_set + [pt]
                if _verify(candidate):
                    current_set = candidate
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_points = list(current_set)
                
                # Early stop if we have 12+
                if best_size >= 12:
                    break
            if best_size >= 12:
                break
        
        if best_size >= 12:
            break
    
    return best_points

def construct():
    """Find a set of integral points in general position with rational distances."""
    points = _find_best_set()
    
    if points is None or len(points) < 3:
        # Fallback: use a known valid small set
        # Points (0,0), (3,0), (0,4) - right triangle, rational distances
        # Add (3,4) - check: dist to (0,0)=5, to (3,0)=4, to (0,4)=3 - all rational
        # But (0,0),(3,0),(0,4),(3,4) - are any 4 concyclic? They form a rectangle, so yes!
        # Need to avoid rectangles.
        # Try: (0,0), (3,0), (0,4), (6,8)
        # dist (0,0)-(6,8) = 10, (3,0)-(6,8) = sqrt(9+64)=sqrt(73) - not rational!
        # Let's use a verified set.
        # Known: 5-point integral point set exists.
        # Use points from literature: 
        # (0,0), (56,0), (0,33), (56,33) - rectangle, bad.
        # Let me just use 3 points that definitely work:
        points = [
            (Fraction(0, 1), Fraction(0, 1)),
            (Fraction(3, 1), Fraction(0, 1)),
            (Fraction(0, 1), Fraction(4, 1)),
        ]
    
    # Convert to required format
    result_points = []
    for x, y in points:
        result_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
    
    return {'D': 1, 'points': result_points}
