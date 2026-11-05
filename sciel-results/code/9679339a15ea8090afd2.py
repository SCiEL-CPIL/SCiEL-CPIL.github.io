"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START

def _fast_greedy_dist_set(start_seed, deadline):
    """
    Optimized greedy algorithm using a set of forbidden squared distances.
    For a candidate point c, it is valid if:
    1. The squared distances from c to all existing points are unique.
    2. None of these squared distances equals the squared distance between any pair of existing points.
    """
    random.seed(start_seed)
    points = []  # List of (x, y) tuples
    points_np = np.empty((0, 2), dtype=np.int32)
    forbidden_dists = set()
    
    # Create list of all possible points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    # Pre-compute all pairs of points? No, we update forbidden_dists dynamically.
    
    for pt in all_points:
        if time.time() > deadline:
            break
        
        if not points:
            # First point is always valid
            points.append(pt)
            points_np = np.array([pt], dtype=np.int32)
            continue
            
        # Vectorized distance calculation
        pt_np = np.array(pt, dtype=np.int32)
        diffs = points_np - pt_np
        d_sq = np.sum(diffs**2, axis=1)
        
        # Check 1: Are all distances from pt to existing points unique?
        # If not, pt forms an isosceles triangle with apex at pt.
        if len(set(d_sq)) != len(d_sq):
            continue
            
        # Check 2: Do any of these distances match a distance between existing points?
        # If d_sq[i] == |p_i - p_j|^2, then p_i, p_j, pt form an isosceles triangle.
        valid = True
        for d in d_sq:
            if d in forbidden_dists:
                valid = False
                break
                
        if valid:
            points.append(pt)
            points_np = np.vstack([points_np, pt_np])
            # Update forbidden distances
            forbidden_dists.update(d_sq)
            
    return points

def _greedy_spiral(deadline):
    """
    Greedy algorithm trying points in a spiral order from the center.
    This tends to find dense clusters.
    """
    points = []
    points_np = np.empty((0, 2), dtype=np.int32)
    forbidden_dists = set()
    
    center = N // 2
    # Generate spiral order
    candidates = []
    for x in range(N):
        for y in range(N):
            dist = (x - center)**2 + (y - center)**2
            candidates.append((dist, x, y))
    candidates.sort()
    
    for dist, x, y in candidates:
        if time.time() > deadline:
            break
        
        pt = (x, y)
        if not points:
            points.append(pt)
            points_np = np.array([pt], dtype=np.int32)
            continue
            
        pt_np = np.array(pt, dtype=np.int32)
        diffs = points_np - pt_np
        d_sq = np.sum(diffs**2, axis=1)
        
        if len(set(d_sq)) != len(d_sq):
            continue
            
        valid = True
        for d in d_sq:
            if d in forbidden_dists:
                valid = False
                break
                
        if valid:
            points.append(pt)
            points_np = np.vstack([points_np, pt_np])
            forbidden_dists.update(d_sq)
            
    return points

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    
    # Strategy 1: Random restarts with the fast greedy algorithm
    # We can run many restarts because the check is O(K) per candidate.
    # Each restart is O(N^2 * K) approx.
    
    # Determine how many restarts we can afford
    # Rough estimate: 4096 candidates * 100 points * 10 ops = 4M ops. ~0.1s in Py/NumPy mix.
    # So ~10-20 restarts per second.
    # 300s * 10 = 3000 restarts? Maybe too optimistic due to Python overhead.
    # Let's just loop until time is up.
    
    seed = 0
    while time.time() < deadline - 5:
        points = _fast_greedy_dist_set(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
            # If we found a very good one, maybe try to improve it?
            # For now, just keep the best.
        seed += 1
        
    # Strategy 2: Spiral order
    if time.time() < deadline - 5:
        points = _greedy_spiral(deadline)
        if len(points) > len(best_points):
            best_points = points
            
    return best_points
# EVOLVE-BLOCK-END
