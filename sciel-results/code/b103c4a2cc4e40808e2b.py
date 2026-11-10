"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from fractions import Fraction
from itertools import combinations

_START = time.time()
_HARD = 115.0  # safety margin vs 150s kill

def _time_ok():
    return (time.time() - _START) < _HARD

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_sq_frac(fr):
    """Check if a Fraction is a perfect square of a rational."""
    if fr <= 0:
        return False
    return _is_square(fr.numerator) and _is_square(fr.denominator)

def _check_collinear(pts):
    """pts: list of (x, y) as Fractions (these are the actual coords x, y*sqrt(D)
    but we work in the 'y' coordinate space; collinearity in the plane uses
    cross product of (x2-x1,y2-y1) and (x3-x1,y3-y1)."""
    for i, j, k in combinations(range(len(pts)), 3):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        cp = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if cp == 0:
            return False
    return True

def _check_concyclic(pts, D):
    """Check no four concyclic. Points are (x, y) where actual y-coord is y*sqrt(D).
    Circle: x^2 + Y^2 + u x + v Y + w = 0 where Y = y*sqrt(D).
    For point (x, y): x^2 + D*y^2 + u*x + v*y*sqrt(D) + w = 0.
    For this to hold with rational u,v,w (in the field), we need the sqrt(D) parts to vanish,
    so v*y must be 0 for each... actually the circle coefficients can be in Q(√D).
    Better: use the determinant condition with the actual squared distances.
    Four points are concyclic iff the determinant of the circle matrix is 0.
    We use coordinates (x, Y) where Y^2 = D*y^2. But Y is irrational generally.
    Alternative exact method: four points P1..P4 in plane are concyclic iff
    the cross ratio condition, or the determinant:
    |x1^2+Y1^2, x1, Y1, 1|
    ... = 0
    Since Y_i = y_i*sqrt(D), Y_i^2 = D*y_i^2 (rational), but Y_i itself is irrational.
    So the matrix has irrational entries. Instead, use the fact that 4 points are concyclic
    iff there exist u,v,w (in Q(sqrt(D))) such that for each point:
    x^2 + D*y^2 + u*x + v*y*sqrt(D) + w = 0
    Write u = u0 + u1*sqrt(D), v = v0 + v1*sqrt(D), w = w0 + w1*sqrt(D).
    Then: x^2 + D*y^2 + (u0+u1*sqrt(D))*x + (v0+v1*sqrt(D))*y*sqrt(D) + (w0+w1*sqrt(D)) = 0
    = [x^2 + D*y^2 + u0*x + v1*D*y + w0] + sqrt(D)*[u1*x + v0*y + w1] = 0
    So both parts must be 0:
    (A) x^2 + D*y^2 + u0*x + v1*D*y + w0 = 0
    (B) u1*x + v0*y + w1 = 0
    This is a linear system in u0,v1,w0,u1,v0,w1 (6 unknowns) with 4 equations (2 per point).
    4 points give 8 equations, 6 unknowns -> overdetermined. They are concyclic iff consistent.
    For 3 points, we can solve for the 6 unknowns (8 eqs, 6 unknowns, still overdetermined but
    3 points always define a circle in this generalized sense? No - 3 points in the plane always
    define a circle, so 3 points are always "concyclic" trivially. The condition is for 4.)
    For 4 points: 8 linear equations in 6 unknowns. Consistent iff the system has a solution.
    We can check by: solve first 3 points' 6 equations for 6 unknowns (need 6 independent eqs from
    3 points = 6 eqs, so exactly determined if the 3 points are non-collinear), then verify 4th point.
    """
    n = len(pts)
    for i, j, k, l in combinations(range(n), 4):
        pts4 = [pts[i], pts[j], pts[k], pts[l]]
        # Build 8x6 matrix (equations in order: A1,B1,A2,B2,A3,B3,A4,B4)
        # Unknowns: u0, v1, w0, u1, v0, w1
        # Eq A for point (x,y): x^2 + D*y^2 + u0*x + v1*D*y + w0 = 0
        #   => x*u0 + D*y*v1 + w0 = -(x^2 + D*y^2)
        # Eq B for point (x,y): u1*x + v0*y + w1 = 0
        #   => x*u1 + y*v0 + w1 = 0
        rows = []
        rhs = []
        for (x, y) in pts4:
            # A eq: u0*x + v1*(D*y) + w0*1 = -(x^2 + D*y^2)
            rows.append([x, Fraction(D) * y, Fraction(1), Fraction(0), Fraction(0), Fraction(0)])
            rhs.append(-(x * x + Fraction(D) * y * y))
            # B eq: u1*x + v0*y + w1*1 = 0
            rows.append([Fraction(0), Fraction(0), Fraction(0), x, y, Fraction(1)])
            rhs.append(Fraction(0))
        # Solve 8x6 system: check consistency via Gaussian elimination
        # Use first 6 rows to solve, verify remaining 2
        M = [row[:] for row in rows[:6]]
        R = rhs[:6]
        # Gaussian elimination on 6x6
        for col in range(6):
            # Find pivot
            pivot = -1
            for r in range(col, 6):
                if M[r][col] != 0:
                    pivot = r
                    break
            if pivot == -1:
                # Singular - this means the 3 points might be degenerate, skip
                break
            if pivot != col:
                M[col], M[pivot] = M[pivot], M[col]
                R[col], R[pivot] = R[pivot], R[col]
            pv = M[col][col]
            for c in range(col, 6):
                M[col][c] /= pv
            R[col] /= pv
            for r in range(6):
                if r != col and M[r][col] != 0:
                    factor = M[r][col]
                    for c in range(col, 6):
                        M[r][c] -= factor * M[col][c]
                    R[r] -= factor * R[col]
        else:
            # Now verify remaining 2 rows
            consistent = True
            for ri in range(6, 8):
                # Compute LHS - should equal rhs[ri]
                lhs = Fraction(0)
                for c in range(6):
                    lhs += rows[ri][c] * R[c]
                if lhs != rhs[ri]:
                    consistent = False
                    break
            if consistent:
                return False  # concyclic found
    return True

def _check_rational_distances(pts, D):
    for i, j in combinations(range(len(pts)), 2):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx * dx + Fraction(D) * dy * dy
        if not _is_sq_frac(d2):
            return False
    return True

def _validate(pts, D):
    if len(pts) < 3:
        return False
    if len(set(pts)) != len(pts):
        return False
    if not _check_rational_distances(pts, D):
        return False
    if not _check_collinear(pts):
        return False
    if not _check_concyclic(pts, D):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + _HARD

    # Known good 5-point configuration (D=1)
    best_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    best_D = 1

    # Strategy: parametric construction on circles
    # For D=1, points (x,y) with rational coords. 
    # Use rational parametrization of a circle to generate many candidate points
    # with guaranteed rational pairwise distances (if on same circle centered at origin).
    
    # For a circle x^2 + y^2 = r^2 with rational r, any two points P,Q on it satisfy
    # |P-Q|^2 = 2r^2 - 2(Px*Qx + Py*Qy). If Px*Qx + Py*Qy is rational, distance is rational.
    # With rational coords, this dot product is always rational. So all points on a rational
    # circle centered at origin have rational pairwise distances.
    
    # We need: no three collinear, no four concyclic.
    # Points on the SAME circle are concyclic (any 4 of them), so we need points on DIFFERENT circles
    # or use a more general construction.
    
    # Better approach: use the known 5-point set and try to add points via search.
    # Also try parametric families.
    
    # Parametric: for D=1, use Pythagorean-like constructions.
    # Points at distance a from origin: (a*cos t, a*sin t) with rational cos t, sin t.
    # Rational points on unit circle: ((1-t^2)/(1+t^2), 2t/(1+t^2)) for rational t.
    
    # Let's try to build a set using points from multiple circles (different radii)
    # to avoid concyclicity, while maintaining rational distances.
    
    # For two points on different circles centered at origin with radii r1, r2:
    # |P-Q|^2 = r1^2 + r2^2 - 2*r1*r2*cos(theta)
    # This is rational iff r1*r2*cos(theta) is rational.
    
    # Alternative: use a lattice-based approach with exact validation.
    # Try to extend the 5-point set by searching for a 6th point.
    
    base_pts = best_points
    
    # Search for a 6th point near the existing configuration
    xs_range = range(-200, 201)
    ys_range = range(0, 201)
    
    found_6 = False
    for x in xs_range:
        if not _time_ok():
            break
        for y in ys_range:
            if not _time_ok():
                break
            new_pt = (Fraction(x), Fraction(y))
            if new_pt in base_pts:
                continue
            candidate = base_pts + [new_pt]
            # Quick checks before full validation
            # Check rational distances to all base points
            ok = True
            for bx, by in base_pts:
                dx = new_pt[0] - bx
                dy = new_pt[1] - by
                d2 = dx*dx + dy*dy
                if not _is_sq_frac(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity
            if not _check_collinear(candidate):
                continue
            # Check concyclicity
            if not _check_concyclic(candidate, 1):
                continue
            best_points = candidate
            best_D = 1
            found_6 = True
            break
        if found_6:
            break
    
    if found_6:
        # Try to add a 7th point
        base_pts = best_points
        for x in range(-300, 301):
            if not _time_ok():
                break
            for y in range(0, 301):
                if not _time_ok():
                    break
                new_pt = (Fraction(x), Fraction(y))
                if new_pt in base_pts:
                    continue
                candidate = base_pts + [new_pt]
                ok = True
                for bx, by in base_pts:
                    dx = new_pt[0] - bx
                    dy = new_pt[1] - by
                    d2 = dx*dx + dy*dy
                    if not _is_sq_frac(d2):
                        ok = False
                        break
                if not ok:
                    continue
                if not _check_collinear(candidate):
                    continue
                if not _check_concyclic(candidate, 1):
                    continue
                best_points = candidate
                best_D = 1
                break
            if len(best_points) > 6:
                break
        if not _time_ok():
            break
    
    # If we didn't find 6+, try a different D with parametric construction
    if len(best_points) <= 5:
        # Try D=2 with parametric points
        # For D=2: distance squared = dx^2 + 2*dy^2
        # We need this to be a perfect square rational for all pairs.
        # Try small integer coordinates
        for D in [2, 3, 5, 6]:
            if not _time_ok():
                break
            # Try to find 5+ points for this D
            pts_D = []
            for x in range(-50, 51):
                if not _time_ok():
                    break
                for y in range(0, 51):
                    if not _time_ok():
                        break
                        pt = (Fraction(x), Fraction(y))
                        if pt in pts_D:
                            continue
                        # Check distances to existing
                        ok = True
                        for px, py in pts_D:
                            dx = pt[0] - px
                            dy = pt[1] - py
                            d2 = dx*dx + Fraction(D)*dy*dy
                            if not _is_sq_frac(d2):
                                ok = False
                                break
                        if not ok:
                            continue
                        # Check collinearity
                        cand = pts_D + [pt]
                        if not _check_collinear(cand):
                            continue
                        # Check concyclicity (only if we have 4+)
                        if len(cand) >= 4 and not _check_concyclic(cand, D):
                            continue
                        pts_D.append(pt)
                        if len(pts_D) >= 6:
                            break
                if len(pts_D) >= 6:
                    break
            if len(pts_D) > len(best_points):
                best_points = pts_D
                best_D = D
    
    # Final validation
    if not _validate(best_points, best_D):
        # Fallback to known good
        best_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                       (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                       (Fraction(16), Fraction(63))]
        best_D = 1
    
    return {'D': best_D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
# EVOLVE-BLOCK-END
