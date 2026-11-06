"""Find maximal integral point sets in general position."""
import time
from itertools import combinations

_START = None

def _elapsed():
    global _START
    if _START is None:
        import time as _t
        _START = _t.time()
    return time.time() - _START

def _deadline_check(limit=130.0):
    return _elapsed() < limit

def _sq(n):
    return n * n

def _is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    while r * r < n:
        r += 1
    while r * r > n:
        r -= 1
    return r * r == n

def _dist_sq(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    # Four points are concyclic iff the determinant is zero
    # |x^2+y^2  x  y  1|
    # Compute using the circle equation: x^2 + y^2 + Dx + Ey + F = 0
    # For each point: x_i^2 + y_i^2 + D*x_i + E*y_i + F = 0
    # Three points determine D, E, F; check if fourth satisfies
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Solve for D, E, F from first three points
    # x1^2+y1^2 + D*x1 + E*y1 + F = 0
    # x2^2+y2^2 + D*x2 + E*y2 + F = 0
    # x3^2+y3^2 + D*x3 + E*y3 + F = 0
    
    # Subtract to eliminate F:
    # D*(x1-x2) + E*(y1-y2) = x2^2+y2^2 - x1^2-y1^2
    # D*(x1-x3) + E*(y1-y3) = x3^2+y3^2 - x1^2-y1^2
    
    a1 = x1 - x2
    b1 = y1 - y2
    c1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
    
    a2 = x1 - x3
    b2 = y1 - y3
    c2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        # Degenerate: first three collinear, can't define circle
        return False
    
    # D = (c1*b2 - c2*b1) / det
    # E = (a1*c2 - a2*c1) / det
    D_num = c1 * b2 - c2 * b1
    E_num = a1 * c2 - a2 * c1
    
    # F = -(x1^2 + y1^2 + D*x1 + E*y1)
    # = -(x1^2 + y1^2) + (D_num*x1 - E_num*y1) / det ... let me just check directly
    # Check point 4: x4^2 + y4^2 + D*x4 + E*y4 + F = 0
    # = x4^2 + y4^2 + D_num/det * x4 + E_num/det * y4 + F
    # F = -(x1^2 + y1^2 + D_num/det * x1 + E_num/det * y1)
    # = -(det*(x1^2+y1^2) + D_num*x1 + E_num*y1) / det
    
    # So check: x4^2+y4^2 + D_num*x4/det + E_num*y4/det - (det*(x1^2+y1^2)+D_num*x1+E_num*y1)/det = 0
    # Multiply by det:
    # det*(x4^2+y4^2) + D_num*x4 + E_num*y4 - det*(x1^2+y1^2) - D_num*x1 - E_num*y1 = 0
    
    return (det * (x4*x4 + y4*y4) + D_num * x4 + E_num * y4 
            - det * (x1*x1 + y1*y1) - D_num * x1 - E_num * y1) == 0

def _valid_set(points):
    """Check all constraints for a set of points."""
    n = len(points)
    # All pairwise distances must be integer
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _dist_sq(points[i], points[j])
            if dsq == 0:
                return False  # duplicate
            if not _is_square(dsq):
                return False
    # No three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    # No four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _check_new_point(pts, new_pt):
    """
    Check if new_pt can be added to pts while maintaining:
    1. Integer distances to all pts
    2. No three collinear (involving new_pt)
    3. No four concyclic (involving new_pt)
    """
    n = len(pts)
    
    # 1. Distance check
    for p in pts:
        dsq = _dist_sq(p, new_pt)
        if dsq == 0:
            return False
        if not _is_square(dsq):
            return False
            
    # 2. Collinearity check: (p_i, p_j, new_pt)
    for i in range(n):
        for j in range(i + 1, n):
            if _are_collinear(pts[i], pts[j], new_pt):
                return False
                
    # 3. Concyclicity check: (p_i, p_j, p_k, new_pt)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_concyclic(pts[i], pts[j], pts[k], new_pt):
                    return False
                    
    return True

def construct():
    global _START
    _START = time.time()
    deadline = 125.0  # safety margin
    
    # Known 5-point solution as fallback
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = fallback
    best_size = 5
    
    # Search for larger configurations
    # Build lattice points and check integrality of distances
    # We'll try a systematic search: fix first few points, extend
    
    # Strategy: use the known 5-point set as seed, try to add a 6th point
    # Also try other seeds from literature
    
    # Known 6-point integral point set? The largest known in the plane is 5
    # Actually, the known maximum for integral point sets in the plane with
    # no three collinear and no four concyclic... let me think.
    # 
    # Actually, there IS a known 6-point integral point set. Let me try to find one.
    # The classic one uses points derived from Pythagorean triples.
    
    # Let me try a targeted search: take the 5-point set and try to find a 6th point
    # that is at integer distance from all 5, not collinear with any pair,
    # and not concyclic with any triple.
    
    seed5 = fallback
    # Try adding a 6th point in a bounded region
    # The 5 points span roughly x: -56..16, y: 0..63
    # Try a wider search
    
    found_6 = None
    search_limit = 150 # Increased limit due to speedup
    
    for x in range(-search_limit, search_limit + 1):
        if not _deadline_check(deadline):
            break
        for y in range(-search_limit, search_limit + 1):
            if not _deadline_check(deadline):
                break
            p = (x, y)
            if _check_new_point(seed5, p):
                found_6 = seed5 + [p]
                best = found_6
                best_size = 6
                break
        if found_6:
            break
    
    # If we found 6, try to find 7
    if best_size == 6:
        seed6 = best
        found_7 = None
        search_limit2 = 120 # Increased limit due to speedup
        for x in range(-search_limit2, search_limit2 + 1):
            if not _deadline_check(deadline):
                break
            for y in range(-search_limit2, search_limit2 + 1):
                if not _deadline_check(deadline):
                    break
                p = (x, y)
                if _check_new_point(seed6, p):
                    found_7 = seed6 + [p]
                    best = found_7
                    best_size = 7
                    break
            if found_7:
                break
    
    # Format result
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': D, 'points': points_out}
