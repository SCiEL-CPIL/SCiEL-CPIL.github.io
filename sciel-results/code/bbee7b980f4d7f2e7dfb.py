"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _is_isosceles(p, q, r):
    """Check if three points form an isosceles triangle (including degenerate)."""
    # Squared distances
    dx1 = p[0] - q[0]
    dy1 = p[1] - q[1]
    d1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = q[0] - r[0]
    dy2 = q[1] - r[1]
    d2_sq = dx2 * dx2 + dy2 * dy2
    
    dx3 = p[0] - r[0]
    dy3 = p[1] - r[1]
    d3_sq = dx3 * dx3 + dy3 * dy3
    
    # Check for any pair of equal distances (isosceles condition)
    # Also check for degenerate cases (collinear points where one is between others)
    # Actually, the problem says "no three points forming an isosceles triangle"
    # Degenerate triangles (collinear) are not triangles, but let's be safe.
    # Standard interpretation: if any two sides are equal, it's isosceles.
    # This includes equilateral and degenerate isosceles.
    if d1_sq == d2_sq or d2_sq == d3_sq or d1_sq == d3_sq:
        return True
    return False

def _check_point_against_set(candidate, points):
    """Check if adding candidate to points creates any isosceles triangle."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _is_isosceles(candidate, points[i], points[j]):
                return False
    return True

def _greedy_random(start_seed, deadline):
    """Greedy algorithm with random ordering."""
    random.seed(start_seed)
    points = []
    # Create list of all possible points and shuffle
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    for pt in all_points:
        if time.time() > deadline:
            break
        if _check_point_against_set(pt, points):
            points.append(pt)
    return points

def _greedy_with_priority(priority_func, deadline):
    """Greedy with a priority function to guide selection."""
    points = []
    # Use a priority queue or sorted list
    # For simplicity, we'll use a list and pick the best available point each time
    # This is O(N^2 * M^2) which might be slow, so let's use a different approach.
    # Actually, let's just use random restarts with different strategies.
    
    # Strategy: Start from center, expand outward
    center = N // 2
    points = []
    
    # Try to add center first
    if _check_point_against_set((center, center), points):
        points.append((center, center))
    
    # Then try points in order of distance from center
    candidates = []
    for x in range(N):
        for y in range(N):
            dist = (x - center) ** 2 + (y - center) ** 2
            candidates.append((dist, x, y))
    candidates.sort()
    
    for dist, x, y in candidates:
        if time.time() > deadline:
            break
        pt = (x, y)
        if pt not in points:
            if _check_point_against_set(pt, points):
                points.append(pt)
    return points

def _fast_greedy_dist_set(start_seed, deadline):
    """
    Fast greedy using the distance-set trick from inspiration 2.
    Maintains a set of all squared distances between existing points.
    When adding a new point P, check:
    1. All distances from P to existing points are distinct.
    2. No distance from P to an existing point is already in the global dist_set.
    If both hold, P is valid.
    """
    random.seed(start_seed)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    points = []
    dist_set = set()  # set of all squared distances between pairs in points
    
    for p in all_points:
        if time.time() > deadline:
            break
        
        px, py = p
        valid = True
        dists_to_existing = []
        
        for i, (qx, qy) in enumerate(points):
            dx = px - qx
            dy = py - qy
            d2 = dx * dx + dy * dy
            dists_to_existing.append((d2, i))
            
            # Condition 2: d(P, i) must not equal any existing pairwise distance
            if d2 in dist_set:
                valid = False
                break
        
        if not valid:
            continue
        
        # Condition 1: All distances from P to existing points must be distinct
        seen_dists = set()
        for d2, i in dists_to_existing:
            if d2 in seen_dists:
                valid = False
                break
            seen_dists.add(d2)
        
        if not valid:
            continue
        
        # Add p to points
        points.append(p)
        
        # Update dist_set with new distances
        for d2, i in dists_to_existing:
            dist_set.add(d2)
    
    return points

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    
    # Try the fast distance-set greedy with multiple random restarts
    for seed in range(50):
        if time.time() > deadline - 5:  # Leave 5s safety margin
            break
        
        points = _fast_greedy_dist_set(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
    
    # Also try the original slower greedy with random restarts
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        
        points = _greedy_random(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
    
    # Also try the center-outward strategy
    if time.time() < deadline - 5:
        points = _greedy_with_priority(None, deadline)
        if len(points) > len(best_points):
            best_points = points
    
    # Final check: ensure the result is valid
    # (The greedy algorithm should guarantee this, but let's be safe)
    # For large sets, full validation is expensive, so we trust the greedy check.
    
    return best_points
# EVOLVE-BLOCK-END
