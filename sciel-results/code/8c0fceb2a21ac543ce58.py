# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

_TIME_LIMIT = 140.0
_START = time.monotonic()

def _elapsed():
    return time.monotonic() - _START

def _deadline_ok():
    return _elapsed() < _TIME_LIMIT

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

def _gcd(a, b):
    a = abs(a); b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _sqfree_part(n):
    """Return (s, r) with n = s * r^2, s squarefree, r>=0, s>=0. For n>=0."""
    if n < 0:
        return None
    s = 1
    r = 1
    x = n
    p = 2
    while p * p <= x:
        while x % (p * p) == 0:
            x //= (p * p)
            r *= p
        while x % p == 0:
            x //= p
            s *= p
        p += 1
    if x > 1:
        s *= x
    return s, r

def _is_sqfree(n):
    if n <= 1:
        return True
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

# Represent a point as (x, y) where the actual 2D coordinate is (x, y*sqrt(D)).
# x, y are Fractions.

def _dist_sq(p1, p2, D):
    """Squared distance between two points (as Fractions), returns a Fraction.
    Actual coords: (x, y*sqrt(D)). dist^2 = (x1-x2)^2 + D*(y1-y2)^2."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + Fraction(D) * dy * dy

def _is_rational_distance(p1, p2, D):
    d2 = _dist_sq(p1, p2, D)
    if d2 < 0:
        return False
    # d2 is a Fraction; distance = sqrt(d2). Need sqrt(d2) rational.
    # d2 = a/b in lowest terms. sqrt(d2) rational iff a and b are perfect squares.
    a = d2.numerator
    b = d2.denominator
    return _is_square(a) and _is_square(b)

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_rational_distance(pts[i], pts[j], D):
                return False
    return True

def _three_collinear(p1, p2, p3):
    """Check if three points are collinear using exact rational arithmetic.
    Points are (x, y) with actual coords (x, y*sqrt(D)).
    Collinear iff the 2D cross product (in actual coords) is zero.
    We work in the (x, y) plane with the transformation; collinearity in actual
    coords is preserved by the linear map (x, y) -> (x, y*sqrt(D)) which is invertible.
    So collinear in (x,y) plane <=> collinear in actual plane."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

def _no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _three_collinear(pts[i], pts[j], pts[k]):
                    return False
    return True

def _circle_coef(p1, p2, p3):
    """Compute circle through 3 points in (x,y) plane (the transformed plane).
    Circle: x^2 + y^2 + A x + B y + C = 0. Returns (A, B, C) as Fractions or None if collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # Linear system:
    # 2 x1 A + 2 y1 B = x1^2 + y1^2 - C  -> x1^2+y1^2 + 2x1 A + 2y1 B + C = 0
    # Solve for A, B, C.
    # Use the standard formula via determinants.
    d = 2 * ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
    if d == 0:
        return None
    a2 = x1 * x1 + y1 * y1
    b2 = x2 * x2 + y2 * y2
    c2 = x3 * x3 + y3 * y3
    A = (a2 * (y2 - y3) + b2 * (y3 - y1) + c2 * (y1 - y2)) / d
    B = (a2 * (x3 - x2) + b2 * (x1 - x3) + c2 * (x2 - x1)) / d
    C = (a2 * (x2 * y3 - x3 * y2) + b2 * (x3 * y1 - x1 * y3) + c2 * (x1 * y2 - x2 * y1)) / d
    return (A, B, C)

def _four_concyclic(p1, p2, p3, p4):
    """Check if 4 points are concyclic (lie on a common circle) in (x,y) plane.
    If any 3 are collinear, they define no circle; in that case, the 4th must also be
    on that line to be 'concyclic' in the degenerate sense. Standard definition: 4 points
    concyclic iff there exists a circle through all 4. If 3 are collinear, no (finite) circle
    passes through them, so 4 points with 3 collinear are NOT concyclic (unless we allow line).
    We treat: concyclic iff the circle through 3 (if it exists) also passes through the 4th.
    If 3 are collinear, return False (not concyclic on a circle)."""
    c = _circle_coef(p1, p2, p3)
    if c is None:
        # 3 collinear -> no circle -> not concyclic
        return False
    A, B, C = c
    x, y = p4
    return x * x + y * y + A * x + B * y + C == 0

def _no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _validate(pts, D):
    """Full validation. Returns True if all constraints hold."""
    if not _all_distances_rational(pts, D):
        return False
    if not _no_three_collinear(pts):
        return False
    if not _no_four_concyclic(pts):
        return False
    return True

def _point_ok(new_pt, pts, D):
    """Check if adding new_pt keeps all constraints (distances, collinearity, concyclicity)."""
    # Distances to all existing
    for p in pts:
        if not _is_rational_distance(new_pt, p, D):
            return False
    # Collinearity: new_pt with any pair
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _three_collinear(new_pt, pts[i], pts[j]):
                return False
    # Concyclicity: new_pt with any triple
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _four_concyclic(new_pt, pts[i], pts[j], pts[k]):
                    return False
    return True

def _seed_points():
    # Known 5-point integer configuration (D=1): (0,0),(-56,33),(-16,63),(0,33),(16,63)
    return [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]

def _try_add_random(pts, D, trials, rng):
    """Try to add a point by random search in the (x,y) rational lattice.
    Search x, y in a bounded box with small denominators."""
    n = len(pts)
    # Bounding box based on existing points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin = min(xs) - 20
    xmax = max(xs) + 20
    ymin = min(ys) - 20
    ymax = max(ys) + 20
    # Try integer points first (denominator 1)
    for _ in range(trials):
        if not _deadline_ok():
            break
        x = Fraction(rng.randint(int(xmin), int(xmax)), 1)
        y = Fraction(rng.randint(int(ymin), int(ymax)), 1)
        new_pt = (x, y)
        # Skip if duplicate
        if any(new_pt[0] == p[0] and new_pt[1] == p[1] for p in pts):
            continue
        if _point_ok(new_pt, pts, D):
            return new_pt
    return None

def _circle_parameterization(p1, p2, p3):
    """Given 3 points, return a function that, given rational t, returns a point on the circle.
    Circle through p1,p2,p3. We parameterize using p1 as base and rational slope.
    Actually use the standard: a circle can be parameterized by a rational point and slope.
    Let's use: pick p1 as a known point. A line through p1 with rational slope m hits the
    circle at p1 and another point. That other point is rational function of m.
    Returns (A,B,C) circle coefficients and p1."""
    c = _circle_coef(p1, p2, p3)
    if c is None:
        return None
    return (c, p1)

def _point_on_circle_from_slope(c, p1, m):
    """Line through p1=(x1,y1) with slope m: y - y1 = m (x - x1).
    Intersect with circle x^2+y^2+A x+B y+C=0. One root is p1. Find the other.
    Substitute y = y1 + m(x-x1).
    x^2 + (y1 + m(x-x1))^2 + A x + B(y1+m(x-x1)) + C = 0.
    This is quadratic in x. We know x1 is one root. Use Vieta to get the other.
    Expand: x^2 + y1^2 + 2 m y1 (x-x1) + m^2 (x-x1)^2 + A x + B y1 + B m (x-x1) + C = 0
    = x^2 + m^2 (x^2 - 2 x1 x + x1^2) + 2 m y1 x - 2 m y1 x1 + A x + B y1 + B m x - B m x1 + y1^2 + C
    Coefficient of x^2: 1 + m^2
    Coefficient of x: -2 m^2 x1 + 2 m y1 + A + B m
    Constant: m^2 x1^2 - 2 m y1 x1 + y1^2 + B y1 - B m x1 + C
    Sum of roots = -(coef x)/(coef x^2). One root is x1, so other root x2 = sum - x1.
    """
    (A, B, C) = c
    x1, y1 = p1
    a2 = 1 + m * m
    a1 = -2 * m * m * x1 + 2 * m * y1 + A + B * m
    a0 = m * m * x1 * x1 - 2 * m * y1 * x1 + y1 * y1 + B * y1 - B * m * x1 + C
    # x1 + x2 = -a1/a2
    x2 = -a1 / a2 - x1
    y2 = y1 + m * (x2 - x1)
    return (x2, y2)

def _try_add_on_circle(pts, D, rng, max_t):
    """Try to add a point by parameterizing circles through triples of existing points."""
    n = len(pts)
    # For each triple, get circle, then try rational slopes m = p/q
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if not _deadline_ok():
                    return None
                data = _circle_parameterization(pts[i], pts[j], pts[k])
                if data is None:
                    continue
                (c, p1) = data
                # Try slopes m in a range
                for q in range(1, max_t):
                    for p in range(-max_t, max_t + 1):
                        m = Fraction(p, q)
                        new_pt = _point_on_circle_from_slope(c, p1, m)
                        # Skip if it's one of the existing points or duplicate
                        if any(new_pt[0] == pt[0] and new_pt[1] == pt[1] for pt in pts):
                            continue
                        if _point_ok(new_pt, pts, D):
                            return new_pt
    return None

def construct():
    global _START, _TIME_LIMIT
    _START = time.monotonic()
    _TIME_LIMIT = 140.0
    rng = random.Random(12345)
    D = 1
    best = _seed_points()
    if not _validate(best, D):
        # Should be valid
        pass

    # Try to grow
    improved = True
    while improved and _deadline_ok():
        improved = False
        # Try circle-based addition
        for attempt in range(200):
            if not _deadline_ok():
                break
            new_pt = _try_add_on_circle(best, D, rng, max_t=12)
            if new_pt is not None:
                best = best + [new_pt]
                improved = True
                break
        if improved:
            continue
        # Try random lattice addition
        for attempt in range(500):
            if not _deadline_ok():
                break
            new_pt = _try_add_random(best, D, trials=1, rng=rng)
            if new_pt is not None:
                best = best + [new_pt]
                improved = True
                break
        if improved:
            continue

    # Safety: ensure valid
    if not _validate(best, D):
        best = _seed_points()

    # Format output
    points_out = []
    for (x, y) in best:
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
