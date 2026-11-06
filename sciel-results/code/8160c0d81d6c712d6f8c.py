"""Optimized program for integral point sets in general position.

Strategy:
1. Use known configurations from literature as starting points.
2. Efficient incremental validation using exact integer arithmetic.
3. Targeted candidate generation: Find points at integer distance from ALL points in a seed set.
4. Backtracking search with pruning on these targeted candidates.
5. Explore multiple seed configurations to maximize the chance of finding a larger set.
"""

import time
from math import isqrt
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _distance_squared(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    return True

def _check_collinear_with_new(pts, new_pt):
    """Check if new_pt is collinear with any two existing points."""
    x3, y3 = new_pt
    for i in range(len(pts)):
        x1, y1 = pts[i]
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                return True
    return False

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
                    a11 = x1 - x2
                    a12 = y1 - y2
                    b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                    a21 = x1 - x3
                    a22 = y1 - y3
                    b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                    det = a11 * a22 - a21 * a12
                    if det == 0:
                        continue
                    A_num = b1 * a22 - b2 * a12
                    B_num = a11 * b2 - a21 * b1
                    C_num = -(x1 * x1 + y1 * y1) * det - x1 * A_num - y1 * B_num
                    lhs = x4 * A_num + y4 * B_num + C_num
                    rhs = -(x4 * x4 + y4 * y4) * det
                    if lhs == rhs:
                        return False
    return True

def _check_concyclic_with_new(pts, new_pt):
    """Check if new_pt is concyclic with any three existing points."""
    x4, y4 = new_pt
    for i in range(len(pts)):
        x1, y1 = pts[i]
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            for k in range(j + 1, len(pts)):
                x3, y3 = pts[k]
                a11 = x1 - x2
                a12 = y1 - y2
                b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                a21 = x1 - x3
                a22 = y1 - y3
                b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                det = a11 * a22 - a21 * a12
                if det == 0:
                    continue
                A_num = b1 * a22 - b2 * a12
                B_num = a11 * b2 - a21 * b1
                C_num = -(x1 * x1 + y1 * y1) * det - x1 * A_num - y1 * B_num
                lhs = x4 * A_num + y4 * B_num + C_num
                rhs = -(x4 * x4 + y4 * y4) * det
                if lhs == rhs:
                    return True
    return False

def _all_distances_integer(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _distance_squared(pts[i], pts[j])
            if not _is_square(d2):
                return False
    return True

def _check_new_point_valid(pts, new_pt):
    """Check if new_pt can be added to pts while maintaining all constraints."""
    if new_pt in pts:
        return False
    # Check all distances to existing points are integers
    for pt in pts:
        d2 = _distance_squared(pt, new_pt)
        if not _is_square(d2):
            return False
    # Check no three collinear
    if _check_collinear_with_new(pts, new_pt):
        return False
    # Check no four concyclic
    if len(pts) >= 3:
        if _check_concyclic_with_new(pts, new_pt):
            return False
    return True

def _validate(pts):
    if len(pts) < 3:
        return False
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    if not _all_distances_integer(pts):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

def _generate_candidates_for_seed(seed_pts, max_coord=200):
    """
    Generate points (x,y) with integer coordinates such that distance to ALL
    points in seed_pts is an integer.
    """
    candidates = set()
    n = len(seed_pts)
    
    # We iterate over a bounding box. 
    # A smarter way is to intersect sets of points at integer distance from each seed point.
    # But for small seeds and bounded coords, direct check is fine if we limit the box.
    # Let's estimate the bounding box from the seed points.
    min_x = min(p[0] for p in seed_pts) - 50
    max_x = max(p[0] for p in seed_pts) + 50
    min_y = min(p[1] for p in seed_pts) - 50
    max_y = max(p[1] for p in seed_pts) + 50
    
    # Expand box if needed, but keep it reasonable to avoid O(N^2) explosion
    # Actually, just checking a grid is O(Box^2 * N).
    # Let's try a slightly larger box around the convex hull.
    # To be safe, we'll search a box of size 150x150 centered on the seed's centroid.
    cx = sum(p[0] for p in seed_pts) // n
    cy = sum(p[1] for p in seed_pts) // n
    radius = 100
    
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            pt = (x, y)
            if pt in seed_pts:
                continue
            valid = True
            for sp in seed_pts:
                d2 = _distance_squared(pt, sp)
                if not _is_square(d2):
                    valid = False
                    break
            if valid:
                candidates.add(pt)
                
    return list(candidates)

def _backtracking_search(initial_pts, candidates, max_depth=10, time_limit=30):
    """Backtracking search to find the largest valid set."""
    start_time = time.time()
    best = list(initial_pts)
    best_len = len(best)
    
    # Sort candidates to find larger sets faster? 
    # Maybe prioritize points that are 'far' from existing points to reduce collinearity/concyclicity risk?
    # Or just keep it simple.
    
    def search(current, remaining, depth):
        nonlocal best, best_len
        if time.time() - start_time > time_limit:
            return
        if len(current) > best_len:
            best = list(current)
            best_len = len(current)
        if depth >= max_depth:
            return
        
        # Optimization: if remaining is empty, return
        if not remaining:
            return

        for i, pt in enumerate(remaining):
            if _check_new_point_valid(current, pt):
                new_remaining = remaining[:i] + remaining[i+1:]
                search(current + [pt], new_remaining, depth + 1)
                if time.time() - start_time > time_limit:
                    return
    
    search(initial_pts, candidates, 0)
    return best

def construct():
    deadline = time.time() + 140  # Safety margin: 140s out of 150s budget
    
    # Known 5-point set
    seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(seed5) if _validate(seed5) else [(0, 0), (1, 0), (0, 1)]
    
    # Strategy 1: Extend the known 5-point set
    # Generate candidates that are at integer distance from all 5 points
    candidates_5 = _generate_candidates_for_seed(seed5, max_coord=200)
    if candidates_5:
        # Limit candidates to avoid combinatorial explosion in backtracking
        # We can't just take the first 100, we should take a diverse set or all if small
        # If the set is large, we might need to prune.
        # For now, if it's < 200, use all. If > 200, we might need a smarter selection.
        # Let's assume the intersection is small.
        if len(candidates_5) > 200:
             # Heuristic: keep those with smallest y? Or just truncate.
             # Truncating might miss the solution. 
             # Let's try to be smarter: group by y? No, just truncate for now.
             candidates_5 = candidates_5[:200]
        
        result_5 = _backtracking_search(seed5, candidates_5, max_depth=8, 
                                        time_limit=min(40, deadline - time.time()))
        if len(result_5) > len(best):
            best = result_5

    # Strategy 2: Search from scratch with base pair (0,0) and (1,0)
    # This explores a different part of the configuration space.
    base_pair = [(0, 0), (1, 0)]
    # Generate candidates at integer distance from (0,0) and (1,0)
    # d1^2 = x^2 + y^2
    # d2^2 = (x-1)^2 + y^2
    # d1^2 - d2^2 = 2x - 1
    # So 2x - 1 must be a difference of two squares.
    # Let's just generate all integer points in a box and filter.
    candidates_base = []
    radius = 150
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if y == 0:
                continue # Points on x-axis are collinear with base
            pt = (x, y)
            d1_sq = x*x + y*y
            d2_sq = (x-1)*(x-1) + y*y
            if _is_square(d1_sq) and _is_square(d2_sq):
                candidates_base.append(pt)
    
    if candidates_base:
        # Limit candidates
        if len(candidates_base) > 300:
            # Sort by y to get diverse heights?
            candidates_base.sort(key=lambda p: abs(p[1]))
            candidates_base = candidates_base[:300]
        
        result_base = _backtracking_search(base_pair, candidates_base, max_depth=8, 
                                           time_limit=min(40, deadline - time.time()))
        if len(result_base) > len(best):
            best = result_base

    # Strategy 3: Try a few other specific seeds if time permits
    # e.g. (0,0) and (3,4)
    if time.time() < deadline - 20 and len(best) < 6:
        base_pair2 = [(0, 0), (3, 4)]
        candidates_base2 = []
        radius = 100
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if (x, y) in base_pair2:
                    continue
                pt = (x, y)
                d1_sq = x*x + y*y
                d2_sq = (x-3)*(x-3) + (y-4)*(y-4)
                if _is_square(d1_sq) and _is_square(d2_sq):
                    candidates_base2.append(pt)
        
        if candidates_base2:
             if len(candidates_base2) > 200:
                 candidates_base2 = candidates_base2[:200]
             result_base2 = _backtracking_search(base_pair2, candidates_base2, max_depth=8, 
                                                 time_limit=min(15, deadline - time.time()))
             if len(result_base2) > len(best):
                 best = result_base2

    # Final validation of the best set found
    if not _validate(best):
        # Fallback to the known valid 5-point set
        best = seed5

    # Format the result
    D = 1
    points_formatted = [[[x, 1], [y, 1]] for x, y in best]
    
    return {'D': D, 'points': points_formatted}

# EVOLVE-BLOCK-END
