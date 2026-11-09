"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_point_valid(pos, points, dist_sets):
    """Check if pos can be added without creating an isosceles triangle."""
    dists_from_new = set()
    dists_list = []
    
    # Check if pos is apex
    for p in points:
        d2 = _sq_dist(pos, p)
        dists_list.append(d2)
        if d2 in dists_from_new:
            return False
        dists_from_new.add(d2)
    
    # Check if any existing point is apex
    for i, d2 in enumerate(dists_list):
        if d2 in dist_sets[i]:
            return False
    
    return True

def _add_point(pos, points, dist_sets):
    """Add a point and update distance sets."""
    dists_from_new = set()
    dists_list = []
    
    for p in points:
        d2 = _sq_dist(pos, p)
        dists_list.append(d2)
        dists_from_new.add(d2)
    
    for i, d2 in enumerate(dists_list):
        dist_sets[i].add(d2)
    points.append(pos)
    dist_sets.append(dists_from_new)

def _remove_point(idx, points, dist_sets):
    """Remove a point and update distance sets."""
    pos = points[idx]
    points.pop(idx)
    removed_dists = dist_sets.pop(idx)
    
    for i in range(len(points)):
        d2 = _sq_dist(pos, points[i])
        dist_sets[i].discard(d2)

def _greedy_construct(timeout, initial_order=None):
    """
    Fast greedy construction with optional initial ordering.
    """
    start_time = time.time()
    best_points = []
    best_dist_sets = []
    
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    
    while time.time() - start_time < timeout:
        current_points = []
        current_dist_sets = []
        
        if initial_order:
            shuffled = initial_order[:]
        else:
            shuffled = all_positions[:]
            random.shuffle(shuffled)
        
        for pos in shuffled:
            if time.time() - start_time > timeout:
                break
            
            if _check_point_valid(pos, current_points, current_dist_sets):
                _add_point(pos, current_points, current_dist_sets)
                
                if len(current_points) > len(best_points):
                    best_points = current_points[:]
                    best_dist_sets = [s.copy() for s in current_dist_sets]

    return best_points, best_dist_sets

def _local_search(points, dist_sets, timeout):
    """
    Local search using swap operations to escape local optima.
    """
    start_time = time.time()
    best_points = points[:]
    best_dist_sets = [s.copy() for s in dist_sets]
    best_size = len(points)
    
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    point_set = set(points)
    
    # Track which positions are in use
    in_use = set(points)
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try to swap each point with a random unused point
        indices = list(range(len(points)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > timeout:
                break
            
            old_pos = points[i]
            
            # Try a few random replacement positions
            for _ in range(50):
                new_pos = random.choice(all_positions)
                if new_pos in in_use:
                    continue
                
                # Check if we can swap: remove old_pos, add new_pos
                # First, temporarily remove old_pos
                _remove_point(i, points, dist_sets)
                
                if _check_point_valid(new_pos, points, dist_sets):
                    _add_point(new_pos, points, dist_sets)
                    in_use.remove(old_pos)
                    in_use.add(new_pos)
                    improved = True
                    break
                else:
                    # Restore old_pos
                    _add_point(old_pos, points, dist_sets)
                    in_use.add(old_pos)
            
            if len(points) > best_size:
                best_size = len(points)
                best_points = points[:]
                best_dist_sets = [s.copy() for s in dist_sets]
        
        if not improved:
            break
    
    return best_points, best_dist_sets

def _sidon_line():
    """Construct points on a line using Sidon-like positions."""
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def _diagonal_construction():
    """Construct points along the main diagonal with spacing."""
    points = []
    used_diffs = set()
    for i in range(0, N, 1):
        # Check if adding (i, i) creates isosceles with existing points
        valid = True
        for j in range(i):
            d2 = (i-j)**2 + (i-j)**2
            if d2 in used_diffs:
                valid = False
                break
        if valid:
            points.append((i, i))
            for j in range(i):
                used_diffs.add((i-j)**2 + (i-j)**2)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Leave safety margin
    
    best_points = []
    best_dist_sets = []
    
    # Try multiple greedy strategies with different initial orderings
    strategies = []
    
    # Strategy 1: Pure random
    strategies.append(None)
    
    # Strategy 2: Diagonal bias - start with diagonal points
    diag_pts = _diagonal_construction()
    if diag_pts:
        remaining = [(x, y) for x in range(N) for y in range(N) if (x, y) not in set(diag_pts)]
        random.shuffle(remaining)
        strategies.append(diag_pts + remaining)
    
    # Strategy 3: Spiral ordering
    spiral = []
    for radius in range(N):
        for x in range(N):
            for y in range(N):
                d = max(abs(x - N//2), abs(y - N//2))
                if d == radius:
                    spiral.append((x, y))
    random.shuffle(spiral)
    strategies.append(spiral)
    
    # Run greedy for each strategy
    for strategy in strategies:
        if time.time() - start_time > deadline - 30:
            break
        
        timeout = min(30, deadline - time.time() - 30)
        if timeout < 5:
            break
        
        pts, ds = _greedy_construct(timeout, strategy)
        if len(pts) > len(best_points):
            best_points = pts
            best_dist_sets = ds
    
    # Local search to improve
    if time.time() - start_time < deadline - 10 and best_points:
        ls_timeout = deadline - time.time() - 5
        if ls_timeout > 5:
            best_points, best_dist_sets = _local_search(best_points, best_dist_sets, ls_timeout)
    
    return best_points

# EVOLVE-BLOCK-END
