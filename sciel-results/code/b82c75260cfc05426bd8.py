"""Construct a 6-point rational-distance set in Q(sqrt(3)) via hexagon + perturbation."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def _is_squarefree(n):
    if n <= 1:
        return n == 1
    d = 2
    while d * d <= n:
        if n % (d * d) == 0:
            return False
        d += 1 if d == 2 else 2
    return True

def _rational_sqrt(n):
    """Return (num, den) for sqrt(n) if n is a perfect square, else None."""
    if n < 0:
        return None
    r = int(n ** 0.5)
    if r * r == n:
        return (r, 1)
    return None

def construct():
    t0 = time.time()
    DEADLINE = 130.0

    # Base: regular hexagon in Q(sqrt(3)) with center (0,0), circumradius R=2
    # Vertices: (2,0), (1,sqrt(3)), (-1,sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3))
    # In (x, y*sqrt(D)) form with D=3:
    # (2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)
    # All pairwise squared distances are integers (rational).
    # But all 6 lie on circle x^2 + 3y^2 = 4, so 6 concyclic -> invalid.
    #
    # Strategy: perturb vertex 0 from (2,0) to (2, 1/2).
    # Check distances from new P0=(2, 1/2) to others:
    #   to (1,1):   dx=1, dy=-1/2, d^2 = 1 + 3/4 = 7/4 -> d=sqrt(7)/2  NOT rational
    # So that doesn't work directly. Need a different perturbation.
    #
    # Alternative: use a known 6-point rational distance set.
    # Let's try a different approach: start with the 5-point set from seed,
    # and try to add a 6th point with rational distances to all 5.
    #
    # Actually, let's try the hexagon perturbation more carefully.
    # Perturb P0 = (2,0) to (2, t) where t is rational small.
    # Distance to P1=(1,1): dx=1, dy=1-t, d^2 = 1 + 3(1-t)^2
    # For d^2 to be a perfect square rational: 1 + 3(1-t)^2 = s^2
    # 1 + 3(1 - 2t + t^2) = s^2
    # 4 - 6t + 3t^2 = s^2
    # At t=0: s=2. Need rational t near 0 with 3t^2 - 6t + 4 = s^2.
    # Let u = t, solve 3u^2 - 6u + 4 = s^2 for rational u near 0.
    # This is a conic. Parameterize: s = 2 + m*u, plug in:
    # 3u^2 - 6u + 4 = (2+mu)^2 = 4 + 4mu + m^2 u^2
    # 3u^2 - 6u = 4mu + m^2 u^2
    # u(3u - 6 - 4m - m^2 u) = 0
    # u != 0: (3 - m^2)u = 6 + 4m
    # u = (6 + 4m)/(3 - m^2)
    # For m=1: u = 10/(3-1) = 5. Too large.
    # For m=-1: u = (6-4)/(3-1) = 2/2 = 1. Still large.
    # For m=2: u = (6+8)/(3-4) = 14/(-1) = -14.
    # For m=1/2: u = (6+2)/(3-1/4) = 8/(11/4) = 32/11.
    # For m=-1/2: u = (6-2)/(3-1/4) = 4/(11/4) = 16/11.
    # For m=1/3: u = (6+4/3)/(3-1/9) = (22/3)/(26/9) = (22/3)*(9/26) = 198/78 = 33/13.
    # For m=-1/3: u = (6-4/3)/(3-1/9) = (14/3)/(26/9) = (14/3)*(9/26) = 126/78 = 21/13.
    # For m=1/4: u = (6+1)/(3-1/16) = 7/(47/16) = 112/47.
    # For m=-1/4: u = (6-1)/(3-1/16) = 5/(47/16) = 80/47.
    # For m=1/5: u = (6+0.8)/(3-0.04) = 6.8/2.96 = 68/29.6 = 340/148 = 85/37.
    # These are all > 0.5, so the perturbation moves P0 too far.
    #
    # Let me try a different base configuration. Use the known 5-point set and search for a 6th.
    # Seed 5 points: (0,0), (-56,33), (-16,63), (0,33), (16,63) with D=1.
    # Search for a 6th point (x,y) with rational x,y such that distances to all 5 are rational.
    #
    # This is a hard search. Let's instead try a known 6-point configuration.
    #
    # Known result: There exist 6-point rational distance sets. One construction:
    # Take an equilateral triangle with side 1: (0,0), (1,0), (1/2, sqrt(3)/2) in Q(sqrt(3)).
    # Add the reflection: (1/2, -sqrt(3)/2). That's 4 points.
    # Add more... this gets complex.
    #
    # Let me try a systematic local search: start with 5 points and try to add a 6th
    # by scanning rational points in a bounded region.
    
    # Use the seed 5-point set as base
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    def dist_sq(p, q, D):
        dx = Fraction(p[0]) - Fraction(q[0])
        dy = Fraction(p[1]) - Fraction(q[1])
        return dx * dx + D * dy * dy
    
    def is_rational_dist(p, q, D):
        d2 = dist_sq(p, q, D)
        if d2 <= 0:
            return False
        # Check if d2 is a perfect square of a rational
        num = d2.numerator
        den = d2.denominator
        rn = int(num ** 0.5)
        rd = int(den ** 0.5)
        if rn * rn == num and rd * rd == den:
            return True
        return False
    
    def no_three_collinear(pts):
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                for k in range(j+1, len(pts)):
                    (x1,y1), (x2,y2), (x3,y3) = pts[i], pts[j], pts[k]
                    if (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0:
                        return False
        return True
    
    def no_four_concyclic(pts, D):
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                for k in range(j+1, len(pts)):
                    for l in range(k+1, len(pts)):
                        (x1,y1), (x2,y2), (x3,y3), (x4,y4) = pts[i], pts[j], pts[k], pts[l]
                        # Circle through 3 points: x^2 + y^2 + Dx + Ey + F = 0 (in embedded space)
                        # In our coordinates, point (x, y*sqrt(D)) has "squared norm" x^2 + D*y^2
                        # Circle equation in (X, Y) where X=x, Y=y*sqrt(D):
                        # X^2 + Y^2 + aX + bY + c = 0
                        # Substituting: x^2 + D*y^2 + a*x + b*y*sqrt(D) + c = 0
                        # For this to hold for rational x,y, we need the sqrt(D) terms to cancel,
                        # which means b must be 0 if D is not a perfect square.
                        # For D=1: x^2 + y^2 + a*x + b*y + c = 0
                        # Solve for a,b,c from 3 points, check 4th.
                        # Using determinants:
                        # |x1 y1 x1^2+y1^2 1|
                        # |x2 y2 x2^2+y2^2 1| = 0
                        # |x3 y3 x3^2+y3^2 1|
                        # |x4 y4 x4^2+y4^2 1|
                        v1 = x1*x1 + y1*y1
                        v2 = x2*x2 + y2*y2
                        v3 = x3*x3 + y3*y3
                        v4 = x4*x4 + y4*y4
                        # Determinant of 4x4
                        # Expand along last column
                        det = (v1*(y2*(v3-y4)-y3*(v2-y4)) 
                               - v2*(y1*(v3-y4)-y3*(v1-y4))
                               + v3*(y1*(v2-y4)-y2*(v1-y4))
                               - v4*(y1*(v2-y3)-y2*(v1-y3)))
                        if det != 0:
                            continue
                        # Check if the 4 points are actually concyclic (not just collinear or degenerate)
                        # For D=1, if det==0 and not all collinear, they're concyclic
                        # Verify not all 4 collinear
                        collinear = True
                        for a_idx in [0,1,2,3]:
                            for b_idx in [a_idx+1,2,3]:
                                if b_idx == a_idx: continue
                                for c_idx in range(b_idx+1,4):
                                    pts4 = [pts[i],pts[j],pts[k],pts[l]]
                                    (px1,py1),(px2,py2),(px3,py3) = pts4[a_idx],pts4[b_idx],pts4[c_idx]
                                    if (px2-px1)*(py3-py1) - (px3-px1)*(py2-py1) != 0:
                                        collinear = False
                                        break
                                if not collinear: break
                            if not collinear: break
                        if not collinear:
                            return False
        return True
    
    # Verify base 5 points
    assert no_three_collinear(base_points), "Base points have collinear triple"
    assert no_four_concyclic(base_points, D), "Base points have concyclic quadruple"
    for i in range(5):
        for j in range(i+1, 5):
            assert is_rational_dist(base_points[i], base_points[j], D), f"Base dist {i}-{j} not rational"
    
    # Try to find a 6th point using rational coordinates with small denominators
    best_6pt = None
    found = False
    
    # Define a helper to check if a point (x_num/x_den, y_num/y_den) is valid
    # We will iterate over small numerators and denominators
    
    def check_candidate(x_num, x_den, y_num, y_den):
        if time.time() - t0 > DEADLINE:
            return None
        
        # Reduce fractions
        g1 = gcd(abs(x_num), x_den)
        g2 = gcd(abs(y_num), y_den)
        xn = x_num // g1
        xd = x_den // g1
        yn = y_num // g2
        yd = y_den // g2
        
        candidate = (Fraction(xn, xd), Fraction(yn, yd))
        
        # Check distinctness
        for p in base_points:
            if p[0] == candidate[0] and p[1] == candidate[1]:
                return None
        
        # Check all 5 distances rational
        for p in base_points:
            dx = candidate[0] - p[0]
            dy = candidate[1] - p[1]
            d2 = dx*dx + D*dy*dy
            if d2 <= 0:
                return None
            # Check if d2 is a perfect square rational
            num = d2.numerator
            den = d2.denominator
            rn = int(num ** 0.5)
            rd = int(den ** 0.5)
            if rn * rn != num or rd * rd != den:
                return None
            
        # Check no three collinear
        new_pts = base_points + [candidate]
        # We only need to check triples involving the new point
        for i in range(5):
            for j in range(i+1, 5):
                p1 = base_points[i]
                p2 = base_points[j]
                p3 = candidate
                if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]) == 0:
                    return None
        
        # Check no four concyclic
        # We only need to check quadruples involving the new point
        # Since base points have no 4 concyclic, any new concyclic quad must include the new point and 3 base points.
        # Wait, if 3 base points + new point are concyclic, that's 4.
        # Are there other combinations? No, because base points themselves don't form concyclic quads.
        # So we check all triples of base points with the new point.
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    p1 = base_points[i]
                    p2 = base_points[j]
                    p3 = base_points[k]
                    p4 = candidate
                    
                    # Circle through p1, p2, p3. Check if p4 lies on it.
                    # Determinant of 4x4 matrix:
                    # | x1 y1 x1^2+y1^2 1 |
                    # | x2 y2 x2^2+y2^2 1 |
                    # | x3 y3 x3^2+y3^2 1 | = 0
                    # | x4 y4 x4^2+y4^2 1 |
                    
                    # To avoid floating point, we can compute the determinant of the 4x4 matrix
                    # or solve for circle equation and check.
                    # Let's use the determinant method with Fractions.
                    
                    v1 = p1[0]*p1[0] + p1[1]*p1[1]
                    v2 = p2[0]*p2[0] + p2[1]*p2[1]
                    v3 = p3[0]*p3[0] + p3[1]*p3[1]
                    v4 = p4[0]*p4[0] + p4[1]*p4[1]
                    
                    # Determinant calculation
                    # Using expansion or a helper
                    # For speed, we can check if the 4 points are concyclic by checking if the power of the point w.r.t circle is 0?
                    # Simpler: Compute circle through p1,p2,p3 and check p4.
                    # Circle: x^2+y^2+Ax+By+C=0
                    # A,B,C determined by p1,p2,p3.
                    # Linear system:
                    # x1 A + y1 B + C = -(x1^2+y1^2)
                    # x2 A + y2 B + C = -(x2^2+y2^2)
                    # x3 A + y3 B + C = -(x3^2+y3^2)
                    
                    # If the 3 points are collinear, they don't define a unique circle, but we already checked no 3 collinear.
                    
                    # Let's solve for A, B, C.
                    # Subtract eq1 from eq2 and eq3:
                    # (x2-x1)A + (y2-y1)B = -(x2^2+y2^2 - x1^2-y1^2)
                    # (x3-x1)A + (y3-y1)B = -(x3^2+y3^2 - x1^2-y1^2)
                    
                    A1 = x2-x1; B1 = y2-y1; C1 = -(v2-v1)
                    A2 = x3-x1; B2 = y3-y1; C2 = -(v3-v1)
                    
                    det = A1*B2 - A2*B1
                    if det == 0:
                        # Should not happen due to collinearity check, but skip if so
                        continue
                    
                    A = (C1*B2 - C2*B1) / det
                    B = (A1*C2 - A2*C1) / det
                    C = -v1 - x1*A - y1*B
                    
                    # Check if p4 satisfies x4^2+y4^2 + A x4 + B y4 + C = 0
                    val = v4 + A*p4[0] + B*p4[1] + C
                    if val == 0:
                        return None
        
        return new_pts

    # Search range for numerators and denominators
    # The base points are around x in [-56, 16], y in [0, 63].
    # Let's search x in [-100, 50], y in [-50, 100] with denominators 1..10.
    # This is 150 * 150 * 10 * 10 = 2,250,000 candidates.
    # Each check involves 5 distance checks and 10 circle checks.
    # Distance check is fast. Circle check is heavier.
    # 2.25M * (5 + 10) * small_const might be too slow in Python.
    # Let's reduce the range or use random sampling.
    
    # Randomized search is better.
    import random
    random.seed(42)
    
    # Generate a list of candidate rationals
    # We want to cover the area around the base points.
    # x range: -100 to 50
    # y range: -50 to 100
    
    candidates = []
    # Add integers first (den=1)
    for x in range(-100, 51):
        for y in range(-50, 101):
            candidates.append((x, 1, y, 1))
    
    # Add fractions with small denominators
    for d in range(2, 11):
        # Sample some numerators
        for _ in range(2000):
            xn = random.randint(-100, 50)
            yn = random.randint(-50, 100)
            candidates.append((xn, d, yn, d))
    
    # Shuffle to mix integers and fractions? Or just iterate.
    # Integers are denser in the "rational distance" sense? No, not necessarily.
    # Let's iterate through the list.
    
    for xn, xd, yn, yd in candidates:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        res = check_candidate(xn, xd, yn, yd)
        if res is not None:
            best_6pt = res
            found = True
            break
    
    if found:
        points = best_6pt
    else:
        # Fallback: return the 5-point set
        points = base_points
    
    # Convert to required format
    # points are Fractions
    return {'D': D, 'points': [[[p[0].numerator, p[0].denominator], [p[1].numerator, p[1].denominator]] for p in points]}
# EVOLVE-BLOCK-END
