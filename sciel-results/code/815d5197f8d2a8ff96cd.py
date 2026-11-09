"""Improved program: hexagon-in-Q(sqrt(3)) with perturbation for 6+ points."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import isqrt, gcd

_DEADLINE = 140.0  # safety margin before 150s kill

def _is_square(n):
    """Check if non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_square_frac(f):
    """Check if a non-negative Fraction is a perfect square of a rational."""
    if f < 0:
        return False
    num, den = f.numerator, f.denominator
    return _is_square(num) and _is_square(den)

def _rational_dist2(p, q, D=1):
    """Squared distance between two points (x, y*sqrt(D)) as a Fraction."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _check_rational_dist(p, q, D=1):
    """Check if distance between p and q is rational."""
    d2 = _rational_dist2(p, q, D)
    return _is_square_frac(d2)

def _collinear(p, q, r):
    """Check if three points are collinear using cross product."""
    # (q-p) x (r-p) = 0
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def _concyclic(p, q, r, s, D=1):
    """Check if four points (x, y*sqrt(D)) are concyclic.
    A circle in this metric: (x-a)^2 + D*(y-b)^2 = R^2
    Expanding: x^2 - 2ax + a^2 + D*y^2 - 2Db*y + Db^2 = R^2
    => x^2 + D*y^2 + (-2a)x + (-2Db)y + (a^2+Db^2-R^2) = 0
    So for point (x,y): x^2 + D*y^2 + A*x + B*y + C = 0
    Four points concyclic iff the 4x4 determinant is zero."""
    def row(pt):
        return (pt[0]*pt[0] + D*pt[1]*pt[1], pt[0], pt[1], Fraction(1))
    A, B, C, Dv = row(p), row(q), row(r), row(s)
    # 4x4 determinant
    det = (A[0]*(B[1]*(C[2]*Dv[3]-C[3]*Dv[2]) - B[2]*(C[1]*Dv[3]-C[3]*Dv[1]) + B[3]*(C[1]*Dv[2]-C[2]*Dv[1]))
         - A[1]*(B[0]*(C[2]*Dv[3]-C[3]*Dv[2]) - B[2]*(C[0]*Dv[3]-C[3]*Dv[0]) + B[3]*(C[0]*Dv[2]-C[2]*Dv[0]))
         + A[2]*(B[0]*(C[1]*Dv[3]-C[3]*Dv[1]) - B[1]*(C[0]*Dv[3]-C[3]*Dv[0]) + B[3]*(C[0]*Dv[1]-C[1]*Dv[0]))
         - A[3]*(B[0]*(C[1]*Dv[2]-C[2]*Dv[1]) - B[1]*(C[0]*Dv[2]-C[2]*Dv[0]) + B[2]*(C[0]*Dv[1]-C[1]*Dv[0])))
    return det == 0

def _check_valid(points, D=1):
    """Full validity check: all pairwise distances rational, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    # Check all pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            if not _check_rational_dist(points[i], points[j], D):
                return False
    # Check no three collinear
    for combo in combinations(range(n), 3):
        if _collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            return False
    # Check no four concyclic
    for combo in combinations(range(n), 4):
        if _concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]], D):
            return False
    return True

def _generate_candidates_near(base_points, D, max_coord=200, max_candidates=5000):
    """Generate candidate rational points with rational distances to all base points."""
    candidates = []
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            pt = (Fraction(x), Fraction(y))
            valid = True
            for bp in base_points:
                if not _check_rational_dist(pt, bp, D):
                    valid = False
                    break
            if valid:
                candidates.append(pt)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    return candidates

def _search_incremental(starting_points, D, max_points=8, time_limit=130.0):
    """Incrementally add points, returning the largest valid set found."""
    start_time = time.time()
    best = list(starting_points)
    current = list(starting_points)
    
    for target_n in range(len(current)+1, max_points+1):
        if time.time() - start_time > time_limit:
            break
        
        # Generate candidates that have rational distance to all current points
        candidates = _generate_candidates_near(current, D, max_coord=150, max_candidates=3000)
        
        # Filter candidates: no collinearity with any pair in current
        filtered = []
        for cand in candidates:
            if cand in current:
                continue
            ok = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if _collinear(cand, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                if len(current) >= 3:
                    for combo in combinations(range(len(current)), 3):
                        if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]], D):
                            ok = False
                            break
            if ok:
                filtered.append(cand)
        
        found = False
        for cand in filtered:
            if time.time() - start_time > time_limit:
                break
            new_set = current + [cand]
            if _check_valid(new_set, D):
                current = new_set
                found = True
                if len(current) > len(best):
                    best = list(current)
                break
        
        if not found:
            candidates2 = _generate_candidates_near(current, D, max_coord=250, max_candidates=5000)
            filtered2 = []
            for cand in candidates2:
                if cand in current:
                    continue
                ok = True
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if _collinear(cand, current[i], current[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok and len(current) >= 3:
                    for combo in combinations(range(len(current)), 3):
                        if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]], D):
                            ok = False
                            break
                if ok:
                    filtered2.append(cand)
            
            for cand in filtered2:
                if time.time() - start_time > time_limit:
                    break
                new_set = current + [cand]
                if _check_valid(new_set, D):
                    current = new_set
                    found = True
                    if len(current) > len(best):
                        best = list(current)
                    break
            
            if not found:
                break
    
    return best

def _build_hexagon_D3():
    """Construct 6 points in Q(sqrt(3)) from regular hexagon vertices.
    Points: (x, y*sqrt(3)) where (x,y) are rational.
    Regular hexagon centered at origin with 'radius' R:
    Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    In standard coords: (R cos(k*pi/3), R sin(k*pi/3))
    For k=0: (R, 0)
    For k=1: (R/2, R*sqrt(3)/2) -> in our coords: (R/2, R/2) since y*sqrt(3) = R*sqrt(3)/2 means y=R/2
    For k=2: (-R/2, R*sqrt(3)/2) -> (-R/2, R/2)
    For k=3: (-R, 0)
    For k=4: (-R/2, -R*sqrt(3)/2) -> (-R/2, -R/2)
    For k=5: (R/2, -R*sqrt(3)/2) -> (R/2, -R/2)
    
    All pairwise distances in this metric:
    dist^2 = (dx)^2 + 3*(dy)^2
    
    Adjacent vertices (e.g., k=0 to k=1): dx=R/2, dy=R/2
    dist^2 = R^2/4 + 3*R^2/4 = R^2 -> dist = R (rational if R rational) ✓
    
    Skip-one (k=0 to k=2): dx=R, dy=R/2
    dist^2 = R^2 + 3*R^2/4 = 7R^2/4 -> dist = R*sqrt(7)/2 (NOT rational!) ✗
    
    Opposite (k=0 to k=3): dx=2R, dy=0
    dist^2 = 4R^2 -> dist = 2R ✓
    
    So regular hexagon does NOT have all rational distances. We need a different approach.
    
    Alternative: Use a known integral point set structure. Let's use the approach of
    taking points on a line and perturbing, or use known small integral point sets.
    
    Actually, let's try a different construction: use points that form a "rational distance"
    configuration. The key insight from the directive is to use the hexagon as a starting
    point and perturb. But since regular hexagon doesn't work, let's try an affine
    transformation or a different base.
    
    Better approach: Start with known valid 5-point sets and search harder.
    Also try D=3 with points designed to have rational distances.
    """
    return None

def _try_D3_construction():
    """Try to build a valid set in Q(sqrt(3)).
    
    In Q(sqrt(3)), distance squared = dx^2 + 3*dy^2.
    We need dx^2 + 3*dy^2 to be a perfect square of a rational for all pairs.
    
    If we set all y=0, it reduces to 1D (collinear, invalid).
    If we use y values such that 3*dy^2 contributes nicely...
    
    Try: points with y in {0, a} for some rational a.
    For two points both with y=0: dist^2 = dx^2 -> need dx rational (always true)
    For two points both with y=a: dist^2 = dx^2 -> need dx rational (always true)  
    For one at y=0, one at y=a: dist^2 = dx^2 + 3a^2 -> need this to be a square
    
    So we need: for each pair (x1, 0) and (x2, a), (x2-x1)^2 + 3a^2 = (rational)^2
    This is equivalent to finding rational right triangles with leg 3a^2... 
    Actually: dx^2 + 3a^2 = s^2, so s^2 - dx^2 = 3a^2, (s-dx)(s+dx) = 3a^2
    
    Let's pick specific values. Set a=1: need dx^2 + 3 = s^2, s^2 - dx^2 = 3
    (s-dx)(s+dx) = 3. Let s-dx = u, s+dx = v, uv=3.
    s = (u+v)/2, dx = (v-u)/2.
    For u=1, v=3: s=2, dx=1. So (0,0) to (1,1) has dist^2 = 1+3=4, dist=2. ✓
    For u=3, v=1: s=2, dx=-1. So (0,0) to (-1,1) has dist^2 = 1+3=4, dist=2. ✓
    
    Let's build a set with some points at y=0 and some at y=1:
    P1=(0,0), P2=(1,1): dist=2 ✓
    P3=(-1,1): dist to P1: 1+3=4→2 ✓, to P2: 4+0=4→2 ✓
    P4=(2,0): dist to P1: 4→2 ✓, to P2: 1+3=4→2 ✓, to P3: 9+3=12→2√3 ✗
    
    Hmm. Let me try more carefully.
    
    Points at y=0: x values must give rational distances to each other (always true, just |dx|)
    Points at y=1: x values must give rational distances to each other (always true, just |dx|)
    Cross pairs: need (x1-x2)^2 + 3 to be a perfect square.
    
    So for a point (a, 0) and (b, 1): (a-b)^2 + 3 must be a square.
    Let d = a-b. d^2 + 3 = s^2. s^2 - d^2 = 3. (s-d)(s+d)=3.
    Rational solutions: s-d = t, s+d = 3/t, so s=(t+3/t)/2, d=(3/t-t)/2.
    
    For t=1: d=1, s=2. So a-b=±1 works.
    For t=3: d=-1, s=2. Same.
    For t=1/3: d=3-t... d=(3/(1/3)-(1/3))/2=(9-1/3)/2=(26/3)/2=13/3, s=(1/3+9)/2=(28/3)/2=14/3
    Check: (13/3)^2 + 3 = 169/9 + 27/9 = 196/9 = (14/3)^2 ✓
    
    So d=13/3 also works! And d=(3/t - t)/2 for any rational t≠0.
    
    Let's build:
    Row 1 (y=0): x = 0, 1, 13/3
    Row 2 (y=1): need each to be at valid d from each row-1 point.
    
    For point (b,1) to be valid with (0,0): b^2+3 must be square → b=±1 or b=±13/3 etc.
    For point (b,1) to be valid with (1,0): (b-1)^2+3 must be square.
    For point (b,1) to be valid with (13/3,0): (b-13/3)^2+3 must be square.
    
    Let b=1: (1-0)^2+3=4=2^2 ✓, (1-1)^2+3=3 ✗. No.
    Let b=-1: 1+3=4 ✓, 4+3=7 ✗. No.
    Let b=13/3: (13/3)^2+3=196/9 ✓, (13/3-1)^2+3=(10/3)^2+3=100/9+27/9=127/9 ✗. No.
    Let b=-13/3: 169/9+27/9=196/9 ✓, (-13/3-1)^2+3=(-16/3)^2+3=256/9+27/9=283/9 ✗. No.
    
    Let's try different row-1 points. Use x=0, x=1, and find a third.
    Need d^2+3 to be square for d=0→3 (no), d=1→4 (yes).
    For third point at x=c: need c^2+3 square AND (c-1)^2+3 square.
    c^2+3=s1^2, (c-1)^2+3=s2^2.
    s1^2-c^2=3, s2^2-(c-1)^2=3.
    
    From s1^2-c^2=3: (s1-c)(s1+c)=3. Let s1-c=u, s1+c=3/u. c=(3/u-u)/2.
    From s2^2-(c-1)^2=3: (s2-(c-1))(s2+(c-1))=3. Let s2-(c-1)=v, s2+(c-1)=3/v. c-1=(3/v-v)/2.
    
    So (3/u-u)/2 - 1 = (3/v-v)/2
    3/u - u - 2 = 3/v - v
    
    Try u=1: c=(3-1)/2=1. Then c=1, but that's our second point.
    Try u=3: c=(1-3)/2=-1. c-1=-2. Need (3/v-v)/2=-2, 3/v-v=-4, 3-v^2=-4v, v^2-4v-3=0, v=(4±√28)/2=2±√7. Not rational.
    Try u=1/3: c=(9-1/3)/2=(26/3)/2=13/3. c-1=10/3. Need (3/v-v)/2=10/3, 3/v-v=20/3, 9-3v^2=20v, 3v^2+20v-9=0, v=(-20±√(400+108))/6=(-20±√508)/6. √508=2√127. Not rational.
    
    This is getting complex. Let me try a simpler approach: just use the existing D=1 search
    but make it much more efficient, and also try D=3 with a smaller, targeted search.
    """
    D = 3
    # Try small grid in Q(sqrt(3))
    best = None
    start_time = time.time()
    
    # Generate candidate points with small integer coordinates
    candidates = []
    for x in range(-20, 21):
        for y in range(-20, 21):
            pt = (Fraction(x), Fraction(y))
            candidates.append(pt)
    
    # Try to build a valid set greedily
    # Start with a promising triple
    # Check: (0,0), (1,1), (-1,1) in D=3
    # dist(0,0)-(1,1) = 1+3=4→2 ✓
    # dist(0,0)-(-1,1) = 1+3=4→2 ✓  
    # dist(1,1)-(-1,1) = 4+0=4→2 ✓
    # Not collinear: cross product of (1,1)-(0,0) and (-1,1)-(0,0) = 1*1-1*(-1)=2≠0 ✓
    
    base = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)), (Fraction(-1), Fraction(1))]
    if not _check_valid(base, D):
        return None
    
    current = list(base)
    best = list(base)
    
    # Try to add points
    for _ in range(5):  # try to add up to 5 more points
        if time.time() - start_time > 60:
            break
        found = False
        for cand in candidates:
            if cand in current:
                continue
            # Quick checks
            ok = True
            for i in range(len(current)):
                if not _check_rational_dist(cand, current[i], D):
                    ok = False
                    break
                for j in range(i):
                    if _collinear(cand, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok and len(current) >= 3:
                for combo in combinations(range(len(current)), 3):
                    if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]], D):
                        ok = False
                        break
            if ok:
                new_set = current + [cand]
                if _check_valid(new_set, D):
                    current = new_set
                    found = True
                    if len(current) > len(best):
                        best = list(current)
                    break
        if not found:
            break
    
    return best, D

def construct():
    start_time = time.time()
    best_points = None
    best_D = 1
    
    # Strategy 1: Try D=3 hexagon-inspired construction
    try:
        result = _try_D3_construction()
        if result is not None:
            pts, D = result
            if pts is not None and len(pts) >= 3:
                best_points = pts
                best_D = D
    except Exception:
        pass
    
    # Strategy 2: D=1 search with known 5-point base
    if best_points is None or len(best_points) < 6:
        try:
            base_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                           (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                           (Fraction(16), Fraction(63))]
            
            if _check_valid(base_points, 1):
                extended = _search_incremental(base_points, 1, max_points=8, time_limit=80.0)
                if best_points is None or len(extended) > len(best_points):
                    best_points = extended
                    best_D = 1
        except Exception:
            pass
    
    # Strategy 3: If still stuck, try D=1 with different base
    if (best_points is None or len(best_points) < 6) and time.time() - start_time < 100:
        try:
            # Try to find a 6-point set in D=1 by searching more broadly
            # Use a different known configuration
            alt_base = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)),
                        (Fraction(0), Fraction(4)), (Fraction(3), Fraction(8)),
                        (Fraction(6), Fraction(4))]
            if _check_valid(alt_base, 1):
                extended_alt = _search_incremental(alt_base, 1, max_points=8, time_limit=50.0)
                if best_points is None or len(extended_alt) > len(best_points):
                    best_points = extended_alt
                    best_D = 1
        except Exception:
            pass
    
    # Fallback: minimal valid set
    if best_points is None:
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))]
        best_D = 1
    
    # Verify final result
    if not _check_valid(best_points, best_D):
        # Emergency fallback
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))]
        best_D = 1
    
    # Format output
    points_out = []
    for pt in best_points:
        x, y = pt
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': points_out}
# EVOLVE-BLOCK-END
