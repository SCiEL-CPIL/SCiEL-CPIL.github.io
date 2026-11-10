"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations
from fractions import Fraction

def is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq_is_square(p1, p2):
    """Check if squared distance between two integer points is a perfect square."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dsq = dx * dx + dy * dy
    return is_perfect_square(dsq)

def check_no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def check_no_four_concyclic(points):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # Four points are concyclic iff the determinant is zero
                    # | x^2+y^2  x  y  1 |
                    # | x1^2+y1^2 x1 y1 1 |
                    # | x2^2+y2^2 x2 y2 1 |
                    # | x3^2+y3^2 x3 y3 1 |
                    # | x4^2+y4^2 x4 y4 1 |
                    # Actually, we can use the condition that the perpendicular bisectors meet at one point
                    # Or use the determinant of the 4x4 matrix
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    # Determinant of:
                    # [a1, x1, y1, 1]
                    # [a2, x2, y2, 1]
                    # [a3, x3, y3, 1]
                    # [a4, x4, y4, 1]
                    # If det == 0, they are concyclic (or collinear, but we already check that)
                    # Compute determinant using expansion
                    # For 4x4:
                    # det = a1*(x2*(y3*1 - y4*3) - y2*(x3*1 - x4*3) + 1*(x3*y4 - x4*y3)) - ...
                    # Let me use a more systematic approach
                    # Subtract row1 from rows 2,3,4 to simplify
                    b2 = a2 - a1
                    c2 = x2 - x1
                    d2 = y2 - y1
                    b3 = a3 - a1
                    c3 = x3 - x1
                    d3 = y3 - y1
                    b4 = a4 - a1
                    c4 = x4 - x1
                    d4 = y4 - y1
                    # Now det of:
                    # [a1, x1, y1, 1]
                    # [b2, c2, d2, 0]
                    # [b3, c3, d3, 0]
                    # [b4, c4, d4, 0]
                    # = 1 * det of 3x3:
                    # [b2, c2, d2]
                    # [b3, c3, d3]
                    # [b4, c4, d4]
                    det3 = b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4)
                    if det3 == 0:
                        return False
    return True

def is_valid_set(points):
    """Check all constraints for a set of integer points."""
    n = len(points)
    if n < 3:
        return False
    # Check all pairwise distances are rational (perfect square for D=1)
    for i in range(n):
        for j in range(i+1, n):
            if not dist_sq_is_square(points[i], points[j]):
                return False
    # Check no three collinear
    if not check_no_three_collinear(points):
        return False
    # Check no four concyclic
    if not check_no_four_concyclic(points):
        return False
    # Check distinct points
    pts_set = set(points)
    if len(pts_set) != n:
        return False
    return True

def try_add_point(current_points, candidate):
    """Try to add candidate to current_points and check validity."""
    new_points = list(current_points) + [candidate]
    return is_valid_set(new_points)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with the known 5-point solution
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_count = len(best_points)
    
    # Generate candidate points in a reasonable range
    # We'll search for points that can be added to the set
    max_coord = 300
    
    # First, try to greedily add points to the existing set
    # Generate all candidate points with integer distances to all current points
    candidates = []
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            if (x, y) in best_points:
                continue
            # Check if distance to all current points is integer
            valid = True
            for px, py in best_points:
                dx = x - px
                dy = y - py
                dsq = dx*dx + dy*dy
                if not is_perfect_square(dsq):
                    valid = False
                    break
            if valid:
                candidates.append((x, y))
    
    # Sort candidates by some heuristic (e.g., distance to origin)
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    # Greedily add points
    current = list(best_points)
    for cand in candidates:
        if time.time() > deadline:
            break
        if try_add_point(current, cand):
            current.append(cand)
            if len(current) > best_count:
                best_count = len(current)
                best_points = list(current)
    
    # Now try local search: move points to create space for more
    # We'll do a simple perturbation search
    import random
    random.seed(42)
    
    # Try to find 6 points by local search
    # Start from the 5-point set and try to add a 6th
    # If we can't, try to rearrange
    
    # Generate a larger pool of candidate points
    all_candidates = []
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            all_candidates.append((x, y))
    
    # Local search: try to find a 6-point set
    # We'll use a simulated annealing-like approach
    # Objective: maximize number of valid points
    
    # For now, let's try a different approach:
    # Search for 6 points directly using a more systematic method
    
    # Known: the problem of finding n points with all pairwise distances integer
    # is related to the "integral point set" problem
    
    # Let me try to find 6 points by searching more carefully
    # We can use the fact that we need all C(6,2)=15 distances to be integers
    
    # Try a randomized search with restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Start from best known
            current = list(best_points)
        else:
            # Random restart with some structure
            current = [(0, 0)]
            # Add random points with integer distances
            for _ in range(4):
                if time.time() > deadline:
                    break
                attempts = 0
                while attempts < 1000:
                    attempts += 1
                    x = random.randint(-max_coord, max_coord)
                    y = random.randint(-max_coord, max_coord)
                    cand = (x, y)
                    if cand in current:
                        continue
                    valid = True
                    for px, py in current:
                        dx = x - px
                        dy = y - py
                        if not is_perfect_square(dx*dx + dy*dy):
                            valid = False
                            break
                    if valid and check_no_three_collinear(current + [cand]):
                        current.append(cand)
                        break
        
        # Try to extend to 6 points
        for cand in all_candidates:
            if time.time() > deadline:
                break
            if cand in current:
                continue
            # Quick check: distance to all current points
            valid = True
            for px, py in current:
                dx = cand[0] - px
                dy = cand[1] - py
                if not is_perfect_square(dx*dx + dy*dy):
                    valid = False
                    break
            if not valid:
                continue
            # Full check
            new_set = current + [cand]
            if is_valid_set(new_set):
                current = new_set
                if len(current) > best_count:
                    best_count = len(current)
                    best_points = list(current)
                break  # Restart the extension
    
    # Final verification
    if is_valid_set(best_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback to known good set
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[-56, 1], [33, 1]], [[-16, 1], [63, 1]], [[0, 1], [33, 1]], [[16, 1], [63, 1]]]}
# EVOLVE-BLOCK-END
