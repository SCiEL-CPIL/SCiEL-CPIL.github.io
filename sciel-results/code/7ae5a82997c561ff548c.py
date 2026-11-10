"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                # Cross product
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    # Four points concyclic iff determinant is zero
                    # | x^2+y^2  x  y  1 |
                    # | x1^2+y1^2 x1 y1 1 |
                    # | x2^2+y2^2 x2 y2 1 |
                    # | x3^2+y3^2 x3 y3 1 |
                    # | x4^2+y4^2 x4 y4 1 |
                    # Use the 4x4 determinant (first row is header)
                    # Actually use rows for points i,j,k,l
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    # Determinant of:
                    # [a1, x1, y1, 1]
                    # [a2, x2, y2, 1]
                    # [a3, x3, y3, 1]
                    # [a4, x4, y4, 1]
                    # Subtract row1 from others to simplify
                    # [a1, x1, y1, 1]
                    # [a2-a1, x2-x1, y2-y1, 0]
                    # [a3-a1, x3-x1, y3-y1, 0]
                    # [a4-a1, x4-x1, y4-y1, 0]
                    # Expand along last column (only first row has 1)
                    # = det of 3x3:
                    # [a2-a1, x2-x1, y2-y1]
                    # [a3-a1, x3-x1, y3-y1]
                    # [a4-a1, x4-x1, y4-y1]
                    b1 = a2 - a1
                    b2 = a3 - a1
                    b3 = a4 - a1
                    c1 = x2 - x1
                    c2 = x3 - x1
                    c3 = x4 - x1
                    d1 = y2 - y1
                    d2 = y3 - y1
                    d3 = y4 - y1
                    det = (b1 * (c2 * d3 - c3 * d2)
                           - c1 * (b2 * d3 - b3 * d2)
                           + d1 * (b2 * c3 - b3 * c2))
                    if det == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2):
                return False
    return True

def _search_integer_points(max_coord=200, max_points=8, deadline=135.0):
    """Search for integer point sets with integer pairwise distances."""
    start_time = time.time()
    
    # Generate candidate points with small integer coordinates
    # Focus on finding sets where all pairwise distances are integers
    # Use Pythagorean-based generation
    
    best_set = None
    best_count = 0
    
    # Start with known 5-point solution
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if _check_no_three_collinear(known_5) and _check_no_four_concyclic(known_5) and _all_distances_rational(known_5, 1):
        best_set = list(known_5)
        best_count = 5
    
    # Try to add more points to the known set
    # For each candidate point, check if it can be added
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    
    # Sort candidates by some heuristic - prefer points that are "far" from existing
    # but have integer distances to many existing points
    def score_candidate(pt, existing):
        count = 0
        for ep in existing:
            dx = pt[0] - ep[0]
            dy = pt[1] - ep[1]
            d2 = dx * dx + dy * dy
            if _is_square(d2):
                count += 1
        return count
    
    # Iteratively try to add points
    current = list(best_set) if best_set else []
    
    for _ in range(200):  # Try up to 200 iterations
        if time.time() > deadline:
            break
        if len(current) >= max_points:
            break
        
        # Find best candidate to add
        best_cand = None
        best_score = -1
        for cand in candidates:
            if cand in current:
                continue
            s = score_candidate(cand, current)
            if s > best_score:
                best_score = s
                best_cand = cand
        
        if best_cand is None or best_score < len(current):
            # Can't add a point that has integer distance to all existing
            # Try a different approach: random restart with different base
            break
        
        # Check if adding best_cand maintains no-three-collinear and no-four-concyclic
        new_set = current + [best_cand]
        if _check_no_three_collinear(new_set) and _check_no_four_concyclic(new_set):
            current = new_set
            if len(current) > best_count:
                best_count = len(current)
                best_set = list(current)
        else:
            # Try next best candidates
            found = False
            for cand in candidates:
                if cand in current or cand == best_cand:
                    continue
                s = score_candidate(cand, current)
                if s < len(current):
                    break
                new_set = current + [cand]
                if _check_no_three_collinear(new_set) and _check_no_four_concyclic(new_set):
                    current = new_set
                    if len(current) > best_count:
                        best_count = len(current)
                        best_set = list(current)
                    found = True
                    break
            if not found:
                break
    
    return best_set if best_set else known_5

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Try to find a better configuration
    best_set = _search_integer_points(max_coord=150, max_points=8, deadline=deadline)
    
    if time.time() > deadline - 5:
        # Fall back to known solution
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the solution
    if not _check_no_three_collinear(best_set) or not _check_no_four_concyclic(best_set) or not _all_distances_rational(best_set, 1):
        # Fall back to known solution
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
