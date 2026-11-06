"""Optimized program for integral point sets in general position.

Strategy:
1. Use known configurations from literature as starting points
2. Efficient incremental validation
3. Backtracking search with pruning
4. Multiple starting configurations to explore different search spaces
"""

import time
from math import isqrt, gcd
from fractions import Fraction
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
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = new_pt
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
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                p4 = new_pt
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
    # Check distinctness
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

def _generate_candidates_from_base(base_pt, max_dist=200):
    """Generate points at integer distance from base_pt."""
    candidates = set()
    for d in range(1, max_dist + 1):
        for dx in range(-d, d + 1):
            dy2 = d * d - dx * dx
            if dy2 < 0:
                continue
            dy = isqrt(dy2)
            if dy * dy != dy2:
                continue
            x = base_pt[0] + dx
            y1 = base_pt[1] + dy
            y2 = base_pt[1] - dy
            if (x, y1) != base_pt:
                candidates.add((x, y1))
            if (x, y2) != base_pt and (x, y2) != (x, y1):
                candidates.add((x, y2))
    return list(candidates)

def _backtracking_search(initial_pts, candidates, max_depth=10, time_limit=30):
    """Backtracking search to find the largest valid set."""
    start_time = time.time()
    best = list(initial_pts)
    best_len = len(best)
    
    def search(current, remaining, depth):
        nonlocal best, best_len
        if time.time() - start_time > time_limit:
            return
        if len(current) > best_len:
            best = list(current)
            best_len = len(current)
        if depth >= max_depth:
            return
        for i, pt in enumerate(remaining):
            if _check_new_point_valid(current, pt):
                new_remaining = remaining[:i] + remaining[i+1:]
                search(current + [pt], new_remaining, depth + 1)
                if time.time() - start_time > time_limit:
                    return
    
    search(initial_pts, candidates, 0)
    return best

def _known_configurations():
    """Return known integral point sets in general position."""
    configs = []
    
    # Seed 5-point set
    seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if _validate(seed5):
        configs.append(seed5)
    
    # Try to find 6-point sets systematically
    # Use the fact that we can start with (0,0) and (a,0) and search for more
    
    return configs

def construct():
    deadline = time.time() + 140  # Safety margin: 140s out of 150s budget
    
    # Start with known 5-point set as baseline
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(seed_points) if _validate(seed_points) else [(0, 0), (1, 0), (0, 1)]
    
    # Strategy: Try multiple starting configurations and use backtracking
    # to find larger sets
    
    # Generate a pool of candidate points
    # Points at integer distance from origin
    candidate_pool = set()
    for d in range(1, 150):
        for dx in range(-d, d + 1):
            dy2 = d * d - dx * dx
            if dy2 < 0:
                continue
            dy = isqrt(dy2)
            if dy * dy != dy2:
                continue
            candidate_pool.add((dx, dy))
            candidate_pool.add((dx, -dy))
    
    candidate_list = list(candidate_pool)
    
    # Try different starting pairs
    start_pairs = [
        [(0, 0), (1, 0)],
        [(0, 0), (3, 4)],
        [(0, 0), (5, 12)],
        [(0, 0), (8, 6)],
        [(0, 0), (15, 8)],
        [(0, 0), (20, 21)],
    ]
    
    # Add some from the seed
    if len(best) >= 2:
        start_pairs.append([best[0], best[1]])
        if len(best) >= 3:
            start_pairs.append([best[0], best[2]])
    
    for start in start_pairs:
        if time.time() > deadline:
            break
        
        # Filter candidates that are at integer distance from both start points
        valid_candidates = []
        for pt in candidate_list:
            if pt in start:
                continue
            if _check_new_point_valid(start, pt):
                valid_candidates.append(pt)
        
        if not valid_candidates:
            continue
        
        # Limit candidates to keep search tractable
        # Sort by "interestingness" - prefer points that don't create collinearity
        valid_candidates = valid_candidates[:200]
        
        # Backtracking search
        result = _backtracking_search(start, valid_candidates, max_depth=8, 
                                      time_limit=min(20, deadline - time.time()))
        
        if len(result) > len(best):
            best = result
    
    # If we still only have the seed, try a more exhaustive search
    if len(best) <= 5 and time.time() < deadline - 30:
        # Try extending the seed set
        seed_candidates = []
        for pt in candidate_list:
            if pt not in best:
                if _check_new_point_valid(best, pt):
                    seed_candidates.append(pt)
        
        if seed_candidates:
            seed_candidates = seed_candidates[:100]
            result = _backtracking_search(best, seed_candidates, max_depth=5,
                                         time_limit=min(25, deadline - time.time()))
            if len(result) > len(best):
                best = result
    
    # Format the result
    D = 1
    points_formatted = [[[x, 1], [y, 1]] for x, y in best]
    
    return {'D': D, 'points': points_formatted}

# EVOLVE-BLOCK-END
