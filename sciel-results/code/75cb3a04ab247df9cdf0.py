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
    Fast greedy construction using O(n) validity check with smart ordering.
    """
    start_time = time.time()
    best_points = []
    
    # Pre-generate all grid points
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    
    # Run multiple random greedy attempts
    while time.time() - start_time < timeout:
        current_points = []
        current_dist_sets = []
        
        # Use different ordering strategies for diversity
        strategy = random.random()
        if strategy < 0.3:
            shuffled = all_positions[:]
            random.shuffle(shuffled)
        elif strategy < 0.5:
            # Spiral from center
            cx, cy = N // 2, N // 2
            shuffled = sorted(all_positions, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        elif strategy < 0.7:
            # Row-major with random row offset
            shuffled = []
            row_offset = random.randint(0, N-1)
            for i in range(N):
                y = (row_offset + i) % N
                row = [(x, y) for x in range(N)]
                if random.random() < 0.5:
                    row.reverse()
                shuffled.extend(row)
        else:
            # Random with slight bias toward corners
            shuffled = all_positions[:]
            random.shuffle(shuffled)
            # Move some corner points to front
            corners = [(0,0), (N-1,0), (0,N-1), (N-1,N-1)]
            shuffled = [c for c in shuffled if c in corners] + [p for p in shuffled if p not in corners]
        
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

def _local_search_improve(points, timeout):
    """
    Try to add more points to an existing set via systematic search.
    """
    start_time = time.time()
    current_points = list(points)
    current_dist_sets = []
    
    # Build distance sets from existing points
    for i, p in enumerate(current_points):
        dists = set()
        for j, q in enumerate(current_points):
            if i != j:
                dists.add(_sq_dist(p, q))
        current_dist_sets.append(dists)
    
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    existing_set = set(current_points)
    candidates = [p for p in all_positions if p not in existing_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for pos in candidates[:]:
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
                candidates.remove(pos)
                improved = True
    
    return current_points

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
    if time_remaining > 10:
        # Split time: 70% for greedy, 30% for local search polish
        greedy_time = (time_remaining - 2) * 0.7
        polish_time = (time_remaining - 2) * 0.3
        
        greedy_pts = _greedy_construct_fast(greedy_time)
        if len(greedy_pts) > len(best_points):
            best_points = greedy_pts
        
        # Local search polish on the best solution
        if time_remaining - (deadline - time.time()) < polish_time and len(best_points) > 0:
            polished = _local_search_improve(best_points, min(polish_time, deadline - time.time()))
            if len(polished) > len(best_points):
                best_points = polished
    
    return best_points

# EVOLVE-BLOCK-END
