"""Rewritten program: construct a larger integral-point set in general position."""
from fractions import Fraction
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Return a set of points (x, y*sqrt(D)) with rational coordinates such that:
    - All pairwise distances are rational.
    - No three points are collinear.
    - No four points are concyclic.
    - D is squarefree, D <= 10^12.
    - |numerators|, denominators <= 10^40.
    Maximize the number of points.
    
    Strategy: Use D=1 (ordinary integer lattice) and search for a large set of
    integer points with all pairwise distances integer, no three collinear, no four
    concyclic. We use a greedy construction with backtracking/local search.
    """
    
    def dist_sq(p, q):
        return (p[0] - q[0])**2 + (p[1] - q[1])**2
    
    def is_integer_dist(p, q):
        d2 = dist_sq(p, q)
        r = isqrt(d2)
        return r * r == d2
    
    def are_collinear(p, q, r):
        # Cross product
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0
    
    def are_concyclic(p, q, r, s):
        # Four points are concyclic iff the determinant is zero
        # |x^2+y^2, x, y, 1| for each point
        def row(pt):
            x, y = pt
            return [x*x + y*y, x, y, 1]
        M = [row(p), row(q), row(r), row(s)]
        # Compute determinant via Gaussian elimination over rationals
        # Use exact arithmetic with Fraction
        n = 4
        M = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
        det = Fraction(1)
        for i in range(n):
            # Find pivot
            pivot_row = -1
            for r in range(i, n):
                if M[r][i] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                return True  # determinant is 0
            if pivot_row != i:
                M[i], M[pivot_row] = M[pivot_row], M[i]
                det = -det
            det *= M[i][i]
            # Eliminate below
            for r in range(i+1, n):
                if M[r][i] != 0:
                    factor = M[r][i] / M[i][i]
                    for c in range(i, n):
                        M[r][c] -= factor * M[i][c]
        return det == 0
    
    def valid_set(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not is_integer_dist(points[i], points[j]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True
    
    # Start with known 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to add more points greedily
    # Candidate points: lattice points within a reasonable bounding box
    # with integer distances to all existing points
    
    def try_add_point(points, candidate):
        """Check if candidate can be added to points."""
        # Check integer distance to all existing
        for p in points:
            if not is_integer_dist(p, candidate):
                return False
        # Check no three collinear
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if are_collinear(points[i], points[j], candidate):
                    return False
        # Check no four concyclic
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if are_concyclic(points[i], points[j], points[k], candidate):
                        return False
        return True
    
    # Search for additional points
    # We'll search in a bounded region around the existing points
    # Use a systematic search over lattice points
    
    best = list(base)
    
    # Generate candidates: points at integer distances from at least 2 existing points
    # For efficiency, we search a bounded box
    min_x = min(p[0] for p in base) - 200
    max_x = max(p[0] for p in base) + 200
    min_y = min(p[1] for p in base) - 200
    max_y = max(p[1] for p in base) + 200
    
    # To make this tractable, we use a smarter approach:
    # For each pair of existing points, find lattice points at integer distance from both
    # This gives a finite candidate set
    
    def find_candidates_from_pair(p, q, limit=300):
        """Find lattice points at integer distance from both p and q, within limit."""
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        d2 = dx*dx + dy*dy
        d = isqrt(d2)
        if d * d != d2:
            return []
        
        # Points at integer distance from p: (p[0] + a, p[1] + b) where a^2+b^2 is a perfect square
        # And also at integer distance from q
        # This is still large, so we bound the search
        candidates = []
        # Search in a box around the midpoint
        mid_x = (p[0] + q[0]) // 2
        mid_y = (p[1] + q[1]) // 2
        
        for x in range(mid_x - limit, mid_x + limit + 1):
            for y in range(mid_y - limit, mid_y + limit + 1):
                cand = (x, y)
                if cand == p or cand == q:
                    continue
                if is_integer_dist(p, cand) and is_integer_dist(q, cand):
                    candidates.append(cand)
        return candidates
    
    # Try to add one more point
    added = False
    for i in range(len(base)):
        for j in range(i+1, len(base)):
            cands = find_candidates_from_pair(base[i], base[j], limit=150)
            for cand in cands:
                if cand in best:
                    continue
                if try_add_point(best, cand):
                    best.append(cand)
                    added = True
                    break
            if added:
                break
        if added:
            break
    
    # Try to add a second more point
    if len(best) > len(base):
        for i in range(len(best)):
            for j in range(i+1, len(best)):
                cands = find_candidates_from_pair(best[i], best[j], limit=150)
                for cand in cands:
                    if cand in best:
                        continue
                    if try_add_point(best, cand):
                        best.append(cand)
                        break
                else:
                    continue
                break
            else:
                continue
            break
    
    # Verify the final set
    if not valid_set(best):
        # Fall back to the base set
        best = list(base)
    
    # Convert to required format
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
