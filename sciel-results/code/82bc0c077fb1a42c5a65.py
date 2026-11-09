"""Integral point sets in general position: maximize number of points."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # We work in the plane with metric: distance^2 between (x1,y1) and (x2,y2)
    # is (x1-x2)^2 + D*(y1-y2)^2. For D=1 this is standard Euclidean.
    # All coordinates are integers (or rationals). We'll search for integer points
    # with D=1 first (simplest), then try small D.

    # Strategy:
    # 1. Precompute a pool of candidate integer points in a bounded box.
    # 2. Use greedy + local search to build a maximal set with all pairwise
    #    distances rational (i.e., squared distances are perfect squares for D=1),
    #    no three collinear, no four concyclic.
    # 3. Try to grow the set via swaps/insertions.

    # For D=1, distance between integer points (x1,y1),(x2,y2) is rational iff
    # (x1-x2)^2 + (y1-y2)^2 is a perfect square.

    # Generate candidate points in a box [-B, B]^2
    # B should be large enough to have many rational-distance pairs.
    # Known: rational distance pairs are dense-ish. Let's use B=200 or so.
    
    # First, let's use the known 5-point set and try to extend it.
    # Known 5-point: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    
    # Let me verify these have integer distances:
    # (0,0)-(-56,33): 56^2+33^2 = 3136+1089 = 4225 = 65^2 ✓
    # (0,0)-(-16,63): 16^2+63^2 = 256+3969 = 4225 = 65^2 ✓
    # (0,0)-(0,33): 33 ✓
    # (0,0)-(16,63): 16^2+63^2 = 4225 = 65^2 ✓
    # (-56,33)-(-16,63): 40^2+30^2 = 1600+900=2500=50^2 ✓
    # (-56,33)-(0,33): 56 ✓
    # (-56,33)-(16,63): 72^2+30^2 = 5184+900=6084=78^2 ✓
    # (-16,63)-(0,33): 16^2+30^2 = 256+900=1156=34^2 ✓
    # (-16,63)-(16,63): 32 ✓
    # (0,33)-(16,63): 16^2+30^2 = 1156=34^2 ✓
    
    # Check no three collinear:
    # (0,0),(-56,33),(-16,63): slopes 33/-56, 63/-16 → not equal ✓
    # (0,0),(-56,33),(0,33): x-coords 0,-56,0 → (0,0) and (0,33) share x=0, but (-56,33) doesn't ✓
    # (0,0),(-56,33),(16,63): slopes 33/-56, 63/16 → not equal ✓
    # (0,0),(-16,63),(0,33): (0,0) and (0,33) share x=0, (-16,63) doesn't ✓
    # (0,0),(-16,63),(16,63): slopes 63/-16, 63/16 → not equal (opposite signs) ✓
    # (0,0),(0,33),(16,63): (0,0) and (0,33) share x=0, (16,63) doesn't ✓
    # (-56,33),(-16,63),(0,33): slopes (63-33)/(-16+56)=30/40=3/4, (33-33)/(0+56)=0 → not equal ✓
    # (-56,33),(-16,63),(16,63): slopes 30/40=3/4, 30/72=5/12 → not equal ✓
    # (-56,33),(0,33),(16,63): slopes 0, 30/16=15/8 → not equal ✓
    # (-16,63),(0,33),(16,63): slopes (33-63)/(0+16)=-30/16=-15/8, (63-33)/(16+16)=30/32=15/16 → not equal ✓
    # (-16,63),(0,33),(16,63) wait, let me recheck: (-16,63) to (0,33): slope=(33-63)/(0-(-16))=-30/16=-15/8
    # (0,33) to (16,63): slope=(63-33)/(16-0)=30/16=15/8. Not equal ✓
    # (0,33),(16,63) and others already checked.
    
    # Check no four concyclic: need to check all C(5,4)=5 subsets.
    # Four points are concyclic iff the determinant of the circle equation vanishes.
    # For points (xi,yi), they're concyclic iff det of matrix with rows [xi^2+yi^2, xi, yi, 1] is 0.
    
    # Let me just proceed with the algorithm.

    def _dist2(p, q, D=1):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx*dx + D*dy*dy

    def _is_rational_dist(p, q, D=1):
        d2 = _dist2(p, q, D)
        return _is_square(d2)

    def _collinear(a, b, c):
        # (b-a) x (c-a) = 0
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

    def _concyclic(a, b, c, d):
        # Four points concyclic iff determinant is zero
        # Using the general circle equation: x^2+y^2+Dx+Ey+F=0 (for D=1, standard)
        # Actually for general D, the "circle" is x^2 + D*y^2 + Ax + By + C = 0
        # Four points are on such a curve iff the 4x4 determinant vanishes.
        def row(p):
            return [p[0]*p[0] + D*p[1]*p[1], p[0], p[1], 1]
        M = [row(a), row(b), row(c), row(d)]
        # Compute determinant
        return _det4(M) == 0

    def _det4(M):
        # 4x4 determinant by cofactor expansion
        def det3(A):
            return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                   -A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                   +A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
        def det2(A):
            return A[0][0]*A[1][1]-A[0][1]*A[1][0]
        d = 0
        for j in range(4):
            minor = []
            for i in range(1,4):
                minor.append([M[i][k] for k in range(4) if k != j])
            d += ((-1)**j) * M[0][j] * det3(minor)
        return d

    def _valid_set(pts, D=1):
        n = len(pts)
        # Check all pairwise distances rational
        for i in range(n):
            for j in range(i+1, n):
                if not _is_rational_dist(pts[i], pts[j], D):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if _collinear(pts[i], pts[j], pts[k]):
                        return False
        # Check no four concyclic
        if n >= 4:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            if _concyclic(pts[i], pts[j], pts[k], pts[l], D):
                                return False
        return True

    def _can_add(pts, new_pt, D=1):
        """Check if new_pt can be added to pts maintaining validity."""
        # Check distance to all existing
        for p in pts:
            if not _is_rational_dist(p, new_pt, D):
                return False
        # Check collinearity with all pairs
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if _collinear(pts[i], pts[j], new_pt):
                    return False
        # Check concyclicity with all triples
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                for k in range(j+1, len(pts)):
                    if _concyclic(pts[i], pts[j], pts[k], new_pt, D):
                        return False
        return True

    # Start with the known 5-point set
    base_pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify base is valid
    assert _valid_set(base_pts, D), "Base set is not valid!"
    
    best_pts = list(base_pts)
    best_n = len(base_pts)
    
    # Generate candidate pool: integer points in a larger box
    # We need points at rational distance from ALL points in the current set.
    # Let's search in a box around the existing points.
    # The existing points span x in [-56,16], y in [0,63].
    # Let's search in [-200, 200] x [0, 200] or so.
    
    # For efficiency, precompute which pairs in the pool have rational distance.
    # But that's O(B^4) which is too much for B=200.
    # Instead, for each candidate, check against current set only.
    
    # Let's try adding points greedily.
    # Search region: expand around the bounding box of current points.
    
    def _try_grow(pts, D, max_search=200, time_limit=None):
        """Try to grow pts by adding one more point."""
        if time_limit is None:
            time_limit = time.time() + 5.0
        
        # Bounding box of current points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs)-max_search, max(xs)+max_search
        ymin, ymax = min(ys)-max_search, max(ys)+max_search
        
        # Try all integer points in the expanded box
        # For speed, iterate and check _can_add
        for x in range(xmin, xmax+1):
            if time.time() > time_limit:
                break
            for y in range(ymin, ymax+1):
                if time.time() > time_limit:
                    break
                cand = (x, y)
                # Skip if already in set
                if cand in pts:
                    continue
                if _can_add(pts, cand, D):
                    return cand
        return None
    
    # Try to grow from 5 to 6
    current = list(base_pts)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        new_pt = _try_grow(current, D, max_search=150, time_limit=time.time()+10.0)
        if new_pt is not None:
            current.append(new_pt)
            best_pts = list(current)
            best_n = len(current)
            # Try to grow further
            continue
        else:
            break
    
    # If we got 6, try to get 7
    if best_n >= 6:
        current = list(best_pts)
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            new_pt = _try_grow(current, D, max_search=200, time_limit=time.time()+10.0)
            if new_pt is not None:
                current.append(new_pt)
                best_pts = list(current)
                best_n = len(current)
                continue
            else:
                break
    
    # Try local search: replace one point to enable adding another
    if time.time() < deadline - 15 and best_n < 8:
        # Try swapping: remove one point, try to add two
        for i in range(len(best_pts)):
            if time.time() > deadline - 10:
                break
            reduced = [best_pts[j] for j in range(len(best_pts)) if j != i]
            # Try to grow reduced by 2
            for step in range(2):
                if time.time() > deadline - 10:
                    break
                new_pt = _try_grow(reduced, D, max_search=150, time_limit=time.time()+8.0)
                if new_pt is not None:
                    reduced.append(new_pt)
                    if len(reduced) > len(best_pts):
                        best_pts = list(reduced)
                        best_n = len(reduced)
    
    # Try D=2,3,5 etc. with small integer coordinates
    for D_try in [2, 3, 5, 7]:
        if time.time() > deadline - 10:
            break
        # For D>1, distance^2 = dx^2 + D*dy^2 must be a perfect square
        # Try small integer points
        base_D = [(0,0), (1,0), (0,1), (1,1)]  # check if valid for this D
        # (0,0)-(1,0): 1 ✓
        # (0,0)-(0,1): D → need D to be perfect square. For D=2,3,5,7: NO!
        # So (0,1) is not at rational distance from (0,0) for non-square D.
        # Need different approach for D>1.
        
        # For D=2: dx^2 + 2*dy^2 = square
        # (0,0) to (a,b): a^2+2b^2 = square
        # b=0: a^2 = square ✓ (any a)
        # b=1: a^2+2 = square → a^2+2=c^2 → c^2-a^2=2 → (c-a)(c+a)=2 → c-a=1,c+a=2 → c=1.5 no
        # b=2: a^2+8=c^2 → c^2-a^2=8 → (c-a)(c+a)=8 → c-a=2,c+a=4 → c=3,a=1 ✓ → (1,2) dist^2=1+8=9=3^2
        # b=3: a^2+18=c^2 → c^2-a^2=18 → (c-a)(c+a)=18 → c-a=2,c+a=9 → c=5.5 no; c-a=3,c+a=6→c=4.5 no
        # b=4: a^2+32=c^2 → c^2-a^2=32 → (c-a)(c+a)=32 → c-a=4,c+a=8→c=6,a=2 ✓ → (2,4) dist^2=4+32=36=6^2
        # b=5: a^2+50=c^2 → c^2-a^2=50 → (c-a)(c+a)=50 → c-a=2,c+a=25→c=13.5 no; c-a=5,c+a=10→c=7.5 no
        # b=6: a^2+72=c^2 → c^2-a^2=72 → (c-a)(c+a)=72 → c-a=6,c+a=12→c=9,a=3 ✓ → (3,6) dist^2=9+72=81=9^2
        # So for D=2, points on line y=0 work, and (1,2),(2,4),(3,6) etc.
        # But we need 2D spread. Let's try:
        # (0,0), (1,0), (2,0), (1,2) 
        # Check (1,0)-(2,0): 1 ✓
        # (1,0)-(1,2): 0+2*4=8 → not square ✗
        
        # This is getting complex. Skip D>1 for now and focus on D=1.
        pass
    
    # Final verification
    if not _valid_set(best_pts, D):
        # Fallback to known valid set
        best_pts = base_pts
        best_n = len(base_pts)
    
    # Format output
    points_out = []
    for (x, y) in best_pts:
        points_out.append([[x, 1], [y, 1]])
    
    return {'D': D, 'points': points_out}

# EVOLVE-BLOCK-END
