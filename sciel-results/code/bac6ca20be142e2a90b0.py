"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _greedy_construct_fast(timeout):
    """
    Fast greedy construction using O(n) validity check.
    """
    start_time = time.time()
    best_points = []
    
    # Pre-generate all grid points
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    
    # Run multiple random greedy attempts
    while time.time() - start_time < timeout:
        current_points = []
        current_dist_sets = []
        
        # Shuffle positions
        shuffled = all_positions[:]
        random.shuffle(shuffled)
        
        for pos in shuffled:
            if time.time() - start_time > timeout:
                break
            
            valid = True
            dists_from_new = set()
            dists_list = []
            
            # Check if pos is apex
            for p in current_points:
                d2 = _sq_dist(pos, p)
                dists_list.append(d2)
                if d2 in dists_from_new:
                    valid = False
                    break
                dists_from_new.add(d2)
            
            # Check if any existing point is apex
            if valid:
                for i, d2 in enumerate(dists_list):
                    if d2 in current_dist_sets[i]:
                        valid = False
                        break
            
            if valid:
                current_points.append(pos)
                new_dists = set(dists_list)
                for i, d2 in enumerate(dists_list):
                    current_dist_sets[i].add(d2)
                current_dist_sets.append(new_dists)
                
                if len(current_points) > len(best_points):
                    best_points = current_points[:]

    return best_points

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

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave safety margin
    
    # Start with deterministic constructions
    candidates = []
    
    # Sidon line
    sidon_pts = _sidon_line()
    candidates.append(sidon_pts)
    
    best_points = max(candidates, key=len)
    
    # Try fast greedy randomized search
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        greedy_pts = _greedy_construct_fast(time_remaining - 2)
        if len(greedy_pts) > len(best_points):
            best_points = greedy_pts
    
    return best_points

# EVOLVE-BLOCK-END
