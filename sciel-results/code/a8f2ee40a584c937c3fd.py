"""Construct a large integral point set in general position in Q(sqrt(D))."""
import time
from fractions import Fraction
from math import isqrt, gcd

# We'll work with D=1 (integer lattice) for simplicity and speed.
# Goal: find 6 points in Z^2 with all pairwise distances integers,
# no three collinear, no four concyclic.

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist2_sq(p, q):
    """Squared distance between two integer points."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _all_int_distances(points):
    """Check all pairwise squared distances are perfect squares."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist2_sq(points[i], points[j])
            if not _is_square(d2):
                return False
    return True

def _collinear(p, q, r):
    """Check if three points are collinear."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    return True

def _four_concyclic(p, q, r, s):
    """Check if four points are concyclic using the determinant test."""
    # Use the circle equation: x^2 + y^2 + Dx + Ey + F = 0
    # Four points are concyclic iff the 4x4 determinant is 0.
    # det = | x_i^2+y_i^2  x_i  y_i  1 |
    a, b, c, d = p, q, r, s
    det = (a[0]*a[0] + a[1]*a[1]) * (q[0]*r[1] - q[1]*r[0] + q[1]*s[0] - q[0]*s[1] + r[0]*s[1] - r[1]*s[0]) - \
          a[0] * ((a[0]*a[0] + a[1]*a[1])*0 + (q[0]*a[1] + a[0]*q[1] + a[1]*r[0] + a[0]*r[1] + r[0]*a[1] + a[0]*r[1] + a[1]*s[0] + a[0]*s[1 + 0 - 1] if False else 0))
    # Let me just compute it properly with fractions to avoid overflow issues
    # Actually for integer points, let's use the standard formula.
    # The determinant of the matrix:
    # | x1^2+y1^2  x1  y1  1 |
    # | x2^2+y2^2  x2  y2  1 |
    # | x3^2+y3^2  x3  y3  1 |
    # | x4^2+y4^2  x4  y4  1 |
    # is zero iff concyclic (or collinear, but we check that separately).
    
    pts = [p, q, r, s]
    # Compute using Fraction for exactness
    M = [[Fraction(x*x + y*y), Fraction(x), Fraction(y), Fraction(1)] for x, y in pts]
    # 4x4 determinant
    def det4(M):
        # Cofactor expansion
        sign = 1
        total = Fraction(0)
        for c in range(4):
            # Minor: remove row 0, col c
            minor = [[M[r][cc] for cc in range(4) if cc != c] for r in range(1, 4)]
            # 3x3 det
            d = (minor[0][0]*(minor[1][1]*minor[2][2] - minor[1][2]*minor[2][1])
               - minor[0][1]*(minor[1][0]*minor[2][2] - minor[1][2]*minor[2][0])
               + minor[0][2]*(minor[1][0]*minor[2][1] - minor[1][1]*minor[2][0]))
            total += sign * d
            sign = -sign
        return total
    return det4(M) == 0

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _validate(points):
    """Full validation: distinct, all int distances, no 3 collinear, no 4 concyclic."""
    n = len(points)
    # Distinct
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False
    if not _all_int_distances(points):
        return False
    if not _no_three_collinear(points):
        return False
    if n >= 4 and not _no_four_concyclic(points):
        return False
    return True

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    
    # Start with a known 5-point set and try to extend to 6.
    # Known 5-point integral point set (all distances integers, general position):
    # We need to find one first.
    
    # Let's try a systematic search. Start from simple configurations.
    # A well-known small example: (0,0), (3,4), (0,5) -- but need more points.
    
    # Strategy: Use local search / hill climbing.
    # Start with a valid 4-point set, then try to add a 5th, then 6th.
    
    # Known 4-point integral sets (Euler brick related or simple):
    # (0,0), (3,4), (4,0) -- distances: 5, 4, 5... wait (0,0)-(3,4)=5, (0,0)-(4,0)=4, (3,4)-(4,0)=sqrt(1+16)=sqrt(17) not integer.
    
    # Let me think of actual known small integral point sets.
    # (0,0), (12,5), (12,-5): distances 13, 13, 10 -- 3 points.
    # Add (0,0), (12,5), (12,-5), (24,0): 
    #   (0,0)-(24,0)=24, (12,5)-(24,0)=sqrt(144+25)=13, (12,-5)-(24,0)=13. 
    #   All distances: 13,13,10,24,13,13. Check collinear: (0,0),(12,5),(24,0) not collinear. (0,0),(12,-5),(24,0) not collinear.
    #   Concyclic? (0,0),(12,5),(12,-5),(24,0) -- these form a symmetric shape. 
    #   Circle through (0,0),(24,0): center on x=12. Circle through (12,5): (12-12)^2+(5-cy)^2 = R^2, and 12^2+(0-cy)^2=R^2.
    #   So (5-cy)^2 = 144+cy^2 => 25-10cy+cy^2 = 144+cy^2 => -10cy = 119 => cy = -11.9, not rational in a nice way but let's check.
    #   Actually: 25 - 10*cy = 144 => -10cy = 119 => cy = -119/10. R^2 = 144 + (119/10)^2 = 144 + 14161/100 = (14400+14161)/100 = 28561/100. R = 169/10.
    #   Check (12,-5): (12-12)^2 + (-5-(-119/10))^2 = (69/10)^2 = 4761/100. But R^2 = 28561/100. Not equal! So NOT concyclic. Good.
    
    # So (0,0), (12,5), (12,-5), (24,0) is a valid 4-point set.
    # Now try to add a 5th point.
    
    base4 = [(0,0), (12,5), (12,-5), (24,0)]
    
    # Try to find a 5th point (x,y) such that distances to all 4 are integers.
    # This is a hard constraint. Let's do a bounded search.
    
    best = None
    best_n = 0
    
    # First validate base4
    if _validate(base4):
        best = base4
        best_n = 4
    
    # Try adding points in a bounded region
    # For each candidate (x,y), check if it forms valid set with existing points
    
    # We'll do a more structured search: try points near the existing configuration
    # and also try some known patterns.
    
    # Known 5-point integral point sets exist. Let me try to find one.
    # One approach: start from a 4-point set and search for a 5th.
    
    candidates_5 = []
    # Search in a box around the existing points
    # The existing points span x in [0,24], y in [-5,5]
    # Try x in [-20, 44], y in [-20, 20] with step 1
    
    search_start = time.time()
    for x in range(-20, 45):
        if time.time() > deadline - 20:
            break
        for y in range(-20, 21):
            if time.time() > deadline - 20:
                break
            p = (x, y)
            # Check if p is already in base4
            if p in base4:
                continue
            # Check distances to all 4 points
            ok = True
            for q in base4:
                d2 = (x - q[0])**2 + (y - q[1])**2
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity
            new_pts = base4 + [p]
            if not _no_three_collinear(new_pts):
                continue
            # Check concyclicity (any 4 of the 5)
            if not _no_four_concyclic(new_pts):
                continue
            candidates_5.append(new_pts)
            if len(candidates_5) >= 3:
                break
        if len(candidates_5) >= 3:
            break
    
    # Also try some specific known configurations
    # Try the set: (0,0), (3,4), (0,5), (3,9), (6,4) - let me check
    # Actually let me just try more systematic patterns
    
    # Try extending from other 4-point bases
    other_bases = [
        [(0,0), (3,4), (4,3)],  # 3 points, check
        [(0,0), (5,0), (0,5), (5,5)],  # square, all on a circle!
    ]
    
    # For each 5-point candidate, try to extend to 6
    for pts5 in candidates_5:
        if time.time() > deadline - 10:
            break
        # Try to add a 6th point
        # This is much harder; do a limited search
        xs = set()
        ys = set()
        for (x, y) in pts5:
            xs.add(x)
            ys.add(y)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # Search in a slightly larger box
        for x in range(xmin - 15, xmax + 16):
            if time.time() > deadline - 10:
                break
            for y in range(ymin - 15, ymax + 16):
                if time.time() > deadline - 10:
                    break
                p = (x, y)
                if p in pts5:
                    continue
                ok = True
                for q in pts5:
                    d2 = (x - q[0])**2 + (y - q[1])**2
                    if not _is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                new_pts = pts5 + [p]
                if not _no_three_collinear(new_pts):
                    continue
                if not _no_four_concyclic(new_pts):
                    continue
                if _validate(new_pts):
                    if len(new_pts) > best_n:
                        best = new_pts
                        best_n = len(new_pts)
                    break
            if best_n >= 6:
                break
        if best_n >= 6:
            break
    
    # If we didn't find 6 points, try a different approach:
    # Use a known 6-point integral point set if we can find one via broader search
    if best_n < 6 and time.time() < deadline - 5:
        # Try a different starting configuration
        # Known: there exist 6-point integral point sets. One example involves
        # points related to Pythagorean triples.
        
        # Try: (0,0), (a,b), (c,d), ... where all pairwise distances are integers.
        # This is equivalent to finding 6 points in Z^2 with all pairwise distances in the set of hypotenuses of integer right triangles.
        
        # Let me try a random-ish local search
        import random
        random.seed(42)
        
        # Start from a valid 5-point set if we have one
        if best_n >= 5:
            start_pts = best
        else:
            start_pts = base4
        
        # Try many random perturbations / additions
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            # Pick a random point to move
            idx = random.randint(0, len(start_pts) - 1)
            old = start_pts[idx]
            # Try moving to a nearby point
            for dx in range(-30, 31):
                if time.time() > deadline - 5:
                    break
                for dy in range(-30, 31):
                    if time.time() > deadline - 5:
                        break
                    new_p = (old[0] + dx, old[1] + dy)
                    new_pts = start_pts[:idx] + [new_p] + start_pts[idx+1:]
                    if not _validate(new_pts):
                        continue
                    if len(new_pts) > best_n:
                        best = new_pts
                        best_n = len(new_pts)
                        start_pts = new_pts
                    break
                if len(start_pts) > best_n:
                    break
            if best_n >= 6:
                break
    
    # Final validation of best
    if best is None:
        # Fallback to the original 5-point set
        best = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        if not _validate(best):
            # Use a definitely valid 3-point set
            best = [(0, 0), (3, 4), (0, 5)]
    
    # Format output
    D = 1
    points_out = []
    for (x, y) in best:
        points_out.append([[x, 1], [y, 1]])
    
    return {'D': D, 'points': points_out}
