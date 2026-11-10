"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Precompute coordinates and squared distances
# Points are indexed by 0 to N*N-1
# point index i corresponds to (i // N, i % N)
num_points = N * N
coords = np.array([(i // N, i % N) for i in range(num_points)])

# dist[i][j] is squared distance between point i and point j
dist = np.zeros((num_points, num_points), dtype=np.int32)

# Compute distances using broadcasting
# coords shape: (num_points, 2)
# diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :] # (num_points, num_points, 2)
# dist = np.sum(diff**2, axis=2)
# But this creates a large temporary array. Let's do it in chunks or just use loops if N is small.
# For N=16, num_points=256. 256x256x2 is small.
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sum(diff**2, axis=2).astype(np.int32)

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle (including degenerate)."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def can_add(p, S_set):
    """Check if point p can be added to set S_set without creating an isosceles triangle."""
    if len(S_set) < 2:
        return True
        
    s_list = list(S_set)
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i + 1, len(s_list)):
            b = s_list[j]
            if is_isosceles(p, a, b):
                return False
    return True

def construct():
    start_time = time.time()
    time_limit = 295  # 5s margin
    
    best_size = num_points
    best_set = None
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # 1. Generate a random maximal set
        # Use a randomized greedy approach
        order = list(range(num_points))
        random.shuffle(order)
        
        current_set = set()
        
        # Greedy add
        for p in order:
            # Check if p can be added
            if can_add(p, current_set):
                current_set.add(p)
        
        # 2. Repair to Maximal
        # Find points that can be added and add them.
        # A single pass is sufficient as reasoned in the original code.
        for p in range(num_points):
            if p not in current_set:
                if can_add(p, current_set):
                    current_set.add(p)
        
        # 3. Local Search: Remove and Refill
        # Try to remove a point and see if we can refill with more points? 
        # No, we want to minimize size. So we want to REMOVE points.
        # But removing a point makes it easier to add others.
        # So we can try to remove a point, then greedily add points, and see if the size is smaller.
        # This is a form of local search.
        
        # Let's implement a simple version:
        # Try to remove a random point, then refill greedily.
        # If the size is smaller, keep the new set.
        
        # We will perform a few rounds of local search on the current set.
        # To keep it fast, we limit the number of removals tried per round.
        
        improved = True
        max_local_steps = 10
        step = 0
        while improved and step < max_local_steps and time.time() - start_time < time_limit:
            improved = False
            step += 1
            
            # Try removing each point in the set
            # We do this in a random order to break symmetry
            removal_candidates = list(current_set)
            random.shuffle(removal_candidates)
            
            for p_remove in removal_candidates:
                if time.time() - start_time > time_limit:
                    break
                
                # Tentatively remove p_remove
                new_set = set(current_set)
                new_set.remove(p_remove)
                
                # Try to add points greedily to new_set
                # We need to find points that can be added.
                # To be efficient, we can just iterate through all points not in new_set.
                # But we want to see if the *size* decreases.
                # Actually, removing one point and refilling might not decrease size if we add more than 1.
                # We want the final maximal set to be smaller.
                
                # Let's just do a full greedy refill on new_set.
                # This is expensive. 
                # Alternative: Just check if we can add *any* point that was previously blocked?
                # No, the goal is to find a smaller maximal set.
                
                # Let's try a simpler heuristic:
                # Remove p_remove.
                # Then, for each point q not in new_set, check if it can be added.
                # If yes, add it.
                # If the final size is smaller than current_size, accept.
                
                # But this is O(N^3) per removal.
                # For N=16, N^3 = 4096. 256 removals * 4096 = 1M ops. Feasible.
                
                # Optimization: We can stop early if size exceeds current_size?
                # No, we need the final maximal size.
                
                # Let's implement the refill.
                # We iterate through all points.
                # To make it faster, we can use a list of points not in new_set.
                # But constructing that list is O(N).
                
                # Let's just iterate through range(num_points).
                for q in range(num_points):
                    if q in new_set:
                        continue
                    if can_add(q, new_set):
                        new_set.add(q)
                
                # Check if new_set is maximal?
                # The greedy refill loop above adds all possible points in a single pass.
                # Is it maximal?
                # Yes, by the same logic as before: if a point could be added, it would have been added
                # when its turn came, unless it was blocked by a point added later.
                # But if it was blocked by a point added later, that point was added, so the blocking point is in the set.
                # Wait, if q is blocked by r, and r is added after q was checked, then q is not added.
                # But r is in the set.
                # Is q still addable? No, because r is in the set.
                # So the set is maximal?
                # Let's re-verify the "single pass maximal" claim.
                # Pass:
                # Check q1. Valid. Add q1.
                # Check q2. Invalid wrt {q1}.
                # End.
                # Set {q1}.
                # Can q2 be added? No, because q1 is there.
                # Is the set maximal?
                # Can any other point be added?
                # We didn't check other points.
                # So we must check ALL points.
                # My loop `for q in range(num_points)` checks all points.
                # So if q was valid at the time of check, it is added.
                # If q was invalid, it remains invalid.
                # So after the pass, no point can be added.
                # Thus, the set is maximal.
                
                if len(new_set) < len(current_set):
                    current_set = new_set
                    improved = True
                    # Restart local search with the new set
                    break
                # If we found a smaller set, we break the removal loop and restart the local search
                # If not, we try the next removal candidate.
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    if best_set is None:
        return []
    
    return [tuple(c) for c in coords[list(best_set)]]
# EVOLVE-BLOCK-END
