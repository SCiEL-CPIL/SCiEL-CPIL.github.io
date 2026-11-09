"""Rewritten program: search for integral point sets in general position."""
import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    t0 = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    def dist2_rational(p, q):
        """Return squared distance as Fraction between two points (x, y*sqrt(D))."""
        D = 1  # we'll use D=1 for simplicity (integer points)
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy

    def is_rational_dist(p, q, D):
        """Check if distance between p and q is rational when points are (x, y*sqrt(D))."""
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        d2 = dx * dx + D * dy * dy
        return _is_square(d2)

    def are_collinear(p1, p2, p3):
        """Check if three points are collinear using cross product."""
        return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) == 0

    def are_concyclic(p1, p2, p3, p4):
        """Check if four points are concyclic using determinant."""
        # Circle equation: x^2 + y^2 + Ax + By + C = 0
        # For point (xi, yi): xi^2 + yi^2 + A*xi + B*yi + C = 0
        # We need the 4x4 determinant to be zero
        def row(p):
            return [p[0]**2 + p[1]**2, p[0], p[1], 1]
        M = [row(p1), row(p2), row(p3), row(p4)]
        # Compute determinant
        def det4(m):
            # Cofactor expansion
            def det3(a):
                return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                        - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                        + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
            return (m[0][0]*det3([m[1][1:], m[2][1:], m[3][1:]])
                    - m[0][1]*det3([m[1][:1]+m[1][2:], m[2][:1]+m[2][2:], m[3][:1]+m[3][2:]])
                    + m[0][2]*det3([m[1][:2]+m[1][3:], m[2][:2]+m[2][3:], m[3][:2]+m[3][3:]])
                    - m[0][3]*det3([m[1][:3], m[2][:3], m[3][:3]]))
        return det4(M) == 0

    def valid_set(points, D):
        """Check all constraints for a set of points."""
        n = len(points)
        if n < 3:
            return True
        # Check all pairwise distances are rational (integer for D=1)
        for i in range(n):
            for j in range(i+1, n):
                if not is_rational_dist(points[i], points[j], D):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def try_add_point(points, D, candidate):
        """Try adding candidate to points, check all constraints."""
        new_pts = points + [candidate]
        return valid_set(new_pts, D)

    # Strategy: Use known small integral point sets and try to extend them
    # Start with known 5-point set and try to find 6th, 7th, etc.
    
    # Known 5-point set (from seed)
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend by searching for new points
    best = base_points[:]
    D = 1
    
    # Verify base is valid
    if not valid_set(best, D):
        # Fallback to simpler sets
        best = [(0, 0), (3, 0), (0, 4)]
        if not valid_set(best, D):
            best = [(0, 0), (1, 0), (0, 1)]
    
    # Search for additional points
    # Generate candidates: integer points in a reasonable range
    # For efficiency, only check candidates that have integer distance to all existing points
    
    max_coord = 200
    found = len(best)
    
    # Precompute for speed: for each existing point, which candidates are at integer distance
    # But that's expensive. Instead, iterate over a grid and check incrementally.
    
    # Better approach: use the fact that if P is at integer distance from all points in S,
    # then P lies on the intersection of circles centered at each point in S.
    
    # For 2 points, intersection of two circles gives at most 2 points.
    # For more points, we need to check.
    
    # Let's do a grid search but with early termination
    # To speed up, first check distance to just the first 2-3 points
    
    def quick_check(candidate, subset):
        """Quick check: is candidate at integer distance from all in subset?"""
        for p in subset:
            dx = candidate[0] - p[0]
            dy = candidate[1] - p[1]
            d2 = dx*dx + dy*dy
            if not _is_square(d2):
                return False
        return True
    
    # Try to extend the best set
    current = best[:]
    
    # Search for points to add
    # Use a smarter search: for each pair of existing points, find points at integer distance from both
    # Then check against the rest
    
    def find_common_integer_distance_points(p1, p2, max_range):
        """Find integer points at integer distance from both p1 and p2."""
        results = []
        d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        if not _is_square(d12):
            return results
        d12_sqrt = isqrt(d12)
        
        # Points at integer distance r1 from p1 and r2 from p2
        # (x-p1x)^2 + (y-p1y)^2 = r1^2
        # (x-p2x)^2 + (y-p2y)^2 = r2^2
        # Subtract: linear equation in x,y
        # This is complex, so just do a bounded grid search
        
        x_min = min(p1[0], p2[0]) - max_range
        x_max = max(p1[0], p2[0]) + max_range
        y_min = min(p1[1], p2[1]) - max_range
        y_max = max(p1[1], p2[1]) + max_range
        
        # Limit the search
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                c = (x, y)
                dx1 = x - p1[0]
                dy1 = y - p1[1]
                d1sq = dx1*dx1 + dy1*dy1
                if not _is_square(d1sq):
                    continue
                dx2 = x - p2[0]
                dy2 = y - p2[1]
                d2sq = dx2*dx2 + dy2*dy2
                if not _is_square(d2sq):
                    continue
                results.append(c)
        return results
    
    # Try extending with small search radius first
    for search_radius in [50, 100, 150, 200]:
        if time.time() - t0 > DEADLINE:
            break
        if len(current) >= 8:
            break
        
        # For each pair in current, find common integer-distance points
        candidates = set()
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                common = find_common_integer_distance_points(current[i], current[j], search_radius)
                for c in common:
                    candidates.add(c)
        
        # Filter candidates: must have integer distance to ALL current points
        valid_candidates = []
        for c in candidates:
            if c in current:
                continue
            if quick_check(c, current):
                valid_candidates.append(c)
        
        # Sort by some heuristic (e.g., distance from centroid)
        if valid_candidates:
            cx = sum(p[0] for p in current) / len(current)
            cy = sum(p[1] for p in current) / len(current)
            valid_candidates.sort(key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
        
        # Try adding candidates one by one
        added_any = True
        while added_any and time.time() - t0 < DEADLINE:
            added_any = False
            for c in valid_candidates:
                if c in current:
                    continue
                # Full check
                new_set = current + [c]
                if valid_set(new_set, D):
                    current = new_set
                    valid_candidates.remove(c)
                    added_any = True
                    break
            if not added_any:
                # Try expanding search
                break
        
        if len(current) > len(best):
            best = current[:]
    
    # Also try some known constructions
    # Try the "Euler" type sets or other known patterns
    
    # Try a different base: start from a smaller set and grow
    # Known 4-point sets that might extend better
    alt_bases = [
        [(0, 0), (3, 0), (0, 4), (3, 4)],  # rectangle - has concyclic issue
        [(0, 0), (5, 0), (0, 12), (5, 12)],  # rectangle
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (3, 4)],
    ]
    
    for base in alt_bases:
        if time.time() - t0 > DEADLINE:
            break
        if not valid_set(base, D):
            continue
        cur = base[:]
        # Try to extend
        for search_radius in [30, 60, 100]:
            if time.time() - t0 > DEADLINE:
                break
            if len(cur) >= len(best) + 1:
                break
            
            candidates = set()
            for i in range(len(cur)):
                for j in range(i+1, len(cur)):
                    common = find_common_integer_distance_points(cur[i], cur[j], search_radius)
                    for c in common:
                        candidates.add(c)
            
            valid_candidates = []
            for c in candidates:
                if c in cur:
                    continue
                if quick_check(c, cur):
                    valid_candidates.append(c)
            
            if valid_candidates:
                cx = sum(p[0] for p in cur) / len(cur)
                cy = sum(p[1] for p in cur) / len(cur)
                valid_candidates.sort(key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
            
            added_any = True
            while added_any and time.time() - t0 < DEADLINE:
                added_any = False
                for c in valid_candidates:
                    if c in cur:
                        continue
                    new_set = cur + [c]
                    if valid_set(new_set, D):
                        cur = new_set
                        valid_candidates.remove(c)
                        added_any = True
                        break
        
        if len(cur) > len(best):
            best = cur[:]
    
    # Final: try to extend the best set further with larger search
    if time.time() - t0 < DEADLINE and len(best) < 8:
        cur = best[:]
        for search_radius in [100, 200, 300, 500]:
            if time.time() - t0 > DEADLINE:
                break
            if len(cur) >= 8:
                break
            
            candidates = set()
            for i in range(len(cur)):
                for j in range(i+1, len(cur)):
                    common = find_common_integer_distance_points(cur[i], cur[j], search_radius)
                    for c in common:
                        candidates.add(c)
            
            valid_candidates = []
            for c in candidates:
                if c in cur:
                    continue
                if quick_check(c, cur):
                    valid_candidates.append(c)
            
            if valid_candidates:
                cx = sum(p[0] for p in cur) / len(cur)
                cy = sum(p[1] for p in cur) / len(cur)
                valid_candidates.sort(key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
            
            added_any = True
            while added_any and time.time() - t0 < DEADLINE:
                added_any = False
                for c in valid_candidates:
                    if c in cur:
                        continue
                    new_set = cur + [c]
                    if valid_set(new_set, D):
                        cur = new_set
                        valid_candidates.remove(c)
                        added_any = True
                        break
        
        if len(cur) > len(best):
            best = cur[:]
    
    # Format result
    result_points = []
    for x, y in best:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': D, 'points': result_points}
