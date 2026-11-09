import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    if isinstance(n, Fraction):
        n = n.numerator
    r = math.isqrt(n)
    return r * r == n

def _dist2_int(p1, p2):
    """Distance squared between two integer points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    """Check if 4 integer points are concyclic using exact integer arithmetic."""
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    a4, b4 = p4
    
    if _are_collinear(p1, p2, p3):
        return False
    
    # Circle through p1, p2, p3: x^2 + y^2 + Dx + Ey + F = 0
    # Using Cramer's rule with exact integers
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - b1
    
    det = x1*y2 - x2*y1
    if det == 0:
        return False
    
    rhs1 = -(a2*a2 - a1*a1 + b2*b2 - b1*b1)
    rhs2 = -(a3*a3 - a1*a1 + b3*b3 - b1*b1)
    
    num_D = rhs1*y2 - rhs2*y1
    num_E = x1*rhs2 - x2*rhs1
    num_F = -((a1*a1 + b1*b1)*det + num_D*a1 + num_E*b1)
    
    # Check p4: a4^2 + b4^2 + D*a4 + E*b4 + F = 0 (multiplied by det)
    check = (a4*a4 + b4*b4)*det + num_D*a4 + num_E*b4 + num_F
    return check == 0

def _validate_point_set(points):
    """Validate a set of integer points: rational distances, no 3 collinear, no 4 concyclic."""
    n = len(points)
    # Check all pairwise distances are perfect squares
    for i in range(n):
        for j in range(i+1, n):
            d2 = _dist2_int(points[i], points[j])
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _try_add_point(base_points, x, y):
    """Check if (x,y) can be added to base_points while maintaining all constraints."""
    new_point = (x, y)
    if new_point in base_points:
        return False
    for p in base_points:
        d2 = _dist2_int(new_point, p)
        if not _is_square(d2):
            return False
    n = len(base_points)
    for i in range(n):
        for j in range(i+1, n):
            if _are_collinear(base_points[i], base_points[j], new_point):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_concyclic(base_points[i], base_points[j], base_points[k], new_point):
                    return False
    return True

def _generate_candidates_for_point(point, max_radius):
    """Generate all integer points at integer distance from a given point."""
    candidates = set()
    for dr in range(1, max_radius + 1):
        for dx in range(-dr, dr + 1):
            dy2 = dr*dr - dx*dx
            if dy2 < 0:
                continue
            dy = math.isqrt(dy2)
            if dy*dy != dy2:
                continue
            candidates.add((point[0] + dx, point[1] + dy))
            if dy != 0:
                candidates.add((point[0] + dx, point[1] - dy))
    return candidates

def _find_common_candidates(points, max_radius):
    """Find points at integer distance from ALL points in the set."""
    if not points:
        return set()
    
    # Start with candidates for the first point
    candidates = _generate_candidates_for_point(points[0], max_radius)
    
    # Intersect with candidates for each other point
    for pt in points[1:]:
        pt_candidates = _generate_candidates_for_point(pt, max_radius)
        candidates = candidates & pt_candidates
    
    # Remove existing points
    for pt in points:
        candidates.discard(pt)
    
    return candidates

def _try_known_6point_sets():
    """Try several known 6-point rational distance sets."""
    # These are configurations where all pairwise distances are integers,
    # and we need to verify no-three-collinear and no-four-concyclic
    known_sets = [
        # Configuration 1: Based on (3,4,5) triangle extended
        [(0, 0), (3, 4), (0, 5), (4, 3), (5, 0), (7, 24)],
        # Configuration 2: Symmetric around origin
        [(-3, 0), (3, 0), (0, 4), (4, 3), (-4, 3), (0, 5)],
        # Configuration 3: Using (5,12,13) and (9,12,15)
        [(0, 0), (5, 12), (9, 12), (12, 5), (15, 0), (12, 12)],
        # Configuration 4: Rectangle-based
        [(0, 0), (3, 4), (5, 0), (5, 4), (0, 5), (3, 5)],
        # Configuration 5: Irregular pentagon + center
        [(0, 0), (7, 24), (24, 7), (25, 0), (24, 24), (7, 7)],
        # Configuration 6: Using multiple Pythagorean triples
        [(0, 0), (8, 15), (15, 8), (17, 0), (15, 15), (8, 8)],
        # Configuration 7: Larger spread
        [(0, 0), (20, 21), (21, 20), (29, 0), (21, 21), (20, 20)],
        # Configuration 8: Asymmetric
        [(0, 0), (5, 0), (0, 5), (5, 5), (3, 4), (4, 7)],
        # Configuration 9: Using (12,5,13)
        [(0, 0), (12, 5), (5, 12), (13, 0), (12, 12), (5, 5)],
        # Configuration 10: Mixed distances
        [(0, 0), (4, 3), (3, 4), (5, 0), (0, 5), (4, 7)],
        # Configuration 11: Wider
        [(-5, 0), (5, 0), (0, 5), (0, -5), (3, 4), (-3, 4)],
        # Configuration 12: Diagonal
        [(0, 0), (5, 12), (12, 5), (13, 0), (13, 13), (0, 13)],
        # Configuration 13: Compact
        [(0, 0), (3, 4), (4, 3), (5, 0), (0, 5), (3, 5)],
        # Configuration 14: Extended
        [(0, 0), (9, 12), (12, 9), (15, 0), (15, 15), (0, 15)],
        # Configuration 15: Star-like
        [(0, 0), (5, 0), (0, 5), (-5, 0), (0, -5), (3, 4)],
        # Configuration 16: L-shaped
        [(0, 0), (3, 4), (5, 0), (5, 4), (8, 4), (8, 0)],
        # Configuration 17: T-shaped
        [(0, 0), (3, 4), (0, 5), (3, 5), (6, 5), (6, 0)],
        # Configuration 18: Using (7,24,25)
        [(0, 0), (7, 24), (24, 7), (25, 0), (25, 25), (0, 25)],
        # Configuration 19: Compact cluster
        [(0, 0), (3, 4), (4, 3), (5, 0), (0, 5), (5, 5)],
        # Configuration 20: Spread out
        [(-12, 0), (12, 0), (0, 12), (5, 12), (-5, 12), (0, 5)],
    ]
    
    for pts in known_sets:
        if _validate_point_set(pts):
            return pts
    return None

def _smart_search_for_extension(base_points, deadline, target_size):
    """Smart search to extend base_points to target_size."""
    current = base_points[:]
    
    while len(current) < target_size:
        found = False
        
        # Try increasing radii
        for max_r in [15, 20, 25, 30, 40, 50, 60, 80, 100]:
            if time.time() > deadline - 2:
                return current
            
            candidates = _find_common_candidates(current, max_r)
            if not candidates:
                continue
            
            # Sort by distance to centroid, prefer points that are far from existing
            cx = sum(p[0] for p in current) / len(current)
            cy = sum(p[1] for p in current) / len(current)
            # Score: prefer points that maximize minimum distance to existing points
            def score(cand):
                min_dist = float('inf')
                for p in current:
                    d = (cand[0]-p[0])**2 + (cand[1]-p[1])**2
                    min_dist = min(min_dist, d)
                return -min_dist  # Negate so we sort ascending (maximize min_dist)
            
            candidates = sorted(candidates, key=score)
            
            for cand in candidates:
                if time.time() > deadline - 1:
                    break
                if _try_add_point(current, cand[0], cand[1]):
                    current.append(cand)
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Known good 5-point base set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate_point_set(base5):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    best_set = base5.copy()
    
    # Strategy 0: Try known 6-point configurations first
    known_6 = _try_known_6point_sets()
    if known_6 is not None:
        best_set = known_6
        # Try to extend to 7, 8 points
        if time.time() < deadline - 15:
            extended = _smart_search_for_extension(best_set, deadline - 5, 8)
            best_set = extended
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
    
    # Strategy 1: Search for 6th point using intersection method
    # Use smaller radii first for speed, then expand
    for max_r in [15, 20, 25, 30, 40, 50, 60, 80, 100, 120]:
        if time.time() > deadline - 10:
            break
        
        candidates = _find_common_candidates(base5, max_r)
        if not candidates:
            continue
        
        # Sort by minimum distance to existing points (maximize spread)
        def score(cand):
            min_dist = float('inf')
            for p in base5:
                d = (cand[0]-p[0])**2 + (cand[1]-p[1])**2
                min_dist = min(min_dist, d)
            return -min_dist
        
        candidates = sorted(candidates, key=score)
        
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            if _try_add_point(base5, cand[0], cand[1]):
                best_set = base5 + [cand]
                # Try to extend further
                if time.time() < deadline - 15:
                    extended = _smart_search_for_extension(best_set, deadline - 5, 8)
                    best_set = extended
                return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
    
    # Strategy 2: Use Pythagorean point generation
    if len(best_set) < 6 and time.time() < deadline - 10:
        # Generate points at integer distance from each base point
        all_candidates = set()
        for pt in best_set:
            all_candidates.update(_generate_candidates_for_point(pt, 50))
        
        # Filter to those at integer distance from all points
        valid_candidates = []
        for cand in all_candidates:
            if cand in best_set:
                continue
            valid = True
            for pt in best_set:
                if not _is_square(_dist2_int(cand, pt)):
                    valid = False
                    break
            if valid:
                valid_candidates.append(cand)
        
        # Sort by distance to centroid
        cx = sum(p[0] for p in best_set) / len(best_set)
        cy = sum(p[1] for p in best_set) / len(best_set)
        valid_candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in valid_candidates:
            if time.time() > deadline - 5:
                break
            if _try_add_point(best_set, cand[0], cand[1]):
                best_set.append(cand)
                if time.time() < deadline - 10:
                    extended = _smart_search_for_extension(best_set, deadline - 5, 8)
                    best_set = extended
                break
    
    # Final attempt: try to extend to maximum
    if len(best_set) >= 5 and time.time() < deadline - 10:
        extended = _smart_search_for_extension(best_set, deadline - 2, 8)
        best_set = extended
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

# EVOLVE-BLOCK-END
