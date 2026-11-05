import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    all_indices = list(range(TOTAL))
    
    # We will use a hybrid approach:
    # 1. Greedy construction with random restarts
    # 2. Local search (2-opt style swap) to improve the set size
    # 3. Local search (add/remove) if possible
    
    # To speed up greedy, we can pre-sort indices by some heuristic?
    # Or just shuffle. Shuffling 4096 items is fast.
    
    # We need a fast way to check if a point can be added.
    # State:
    #   selected: list of indices
    #   point_dists[i]: set of squared distances from selected[i] to other selected points
    #   
    # Check for adding p:
    # 1. For each i in selected:
    #      d = dist2[p, selected[i]]
    #      if d in point_dists[i]: conflict (isosceles with base at selected[i])
    # 2. The set of distances {dist2[p, s] for s in selected} must be unique
    #    i.e., no two selected points are at the same distance from p.
    #    This can be checked by:
    #    dvals = [dist2[p, s] for s in selected]
    #    if len(set(dvals)) < len(dvals): conflict (isosceles with apex at p)
    
    # Optimization:
    # We can maintain a global structure?
    # Since N=64, TOTAL=4096, and set size K is likely < 100.
    # Checking addition is O(K) or O(K log K).
    # Greedy loop is O(TOTAL * K).
    # 4096 * 100 = 409,600 operations per restart.
    # With 1000 restarts, ~400M ops. Python might be too slow for 300s.
    # We need to be efficient.
    
    # Let's use NumPy for the distance checks inside the greedy loop where possible?
    # dist2[p, selected_array]
    # But selected_array is dynamic.
    
    # Let's try to optimize the inner loop.
    
    # Strategy:
    # Run greedy for a certain number of restarts.
    # Keep track of best.
    # Then perform local search on the best solution.
    
    # For local search, we want to maximize size.
    # Move: Add a point if possible.
    # Move: Swap a point with another to allow adding more?
    # Actually, just "Add" is sufficient if we can find a point that fits.
    # If we get stuck, we might need to remove a point to allow others.
    # "Remove-Add" move:
    # 1. Pick a point r to remove.
    # 2. Try to add points that were previously blocked by r.
    # This is complex.
    
    # Let's stick to simple greedy first, but optimized.
    
    # Optimization 1: Precompute for each point, the list of points at each distance?
    # No, that's huge.
    
    # Optimization 2: Use bitsets?
    # Python sets are okay.
    
    # Let's estimate time for one greedy run.
    # 4096 iterations.
    # Each iteration:
    #   Check against K points.
    #   K grows from 0 to ~80.
    #   Avg K ~ 40.
    #   4096 * 40 = 163,840 distance lookups and set checks.
    #   In Python, a set lookup is ~50ns? No, more like 100-200ns.
    #   163,840 * 200ns = 0.03 seconds.
    #   So 1000 restarts = 30 seconds.
    #   This is feasible.
    
    # Let's implement the greedy.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_indices)
        
        selected = []
        point_dists = [] # list of sets
        
        for p in all_indices:
            # Check if p can be added
            valid = True
            
            if selected:
                # Get distances from p to all selected
                # selected is a list of ints
                # dist2 is a numpy array
                # We can do: dvals = dist2[p, selected]
                # But selected is a list, so this creates a new array each time?
                # Let's try.
                
                # Check 1: Distances from p to selected must be unique
                # Check 2: For each s in selected, dist(p,s) not in point_dists[index_of_s]
                
                # Efficient check:
                # 1. Get array of distances
                # dvals = dist2[p, selected]  # This is a numpy array slice/lookup
                # 2. Check uniqueness:
                # np.unique(dvals).size == len(dvals)
                # 3. Check conflict with existing isosceles bases:
                #    For each i, dvals[i] not in point_dists[i]
                
                # This requires iterating i anyway for step 3.
                # So we can do it in one loop.
                
                # But wait, if we use numpy for step 2, we avoid a Python loop for uniqueness.
                # Step 3 still requires a loop.
                
                # Let's try:
                dvals = dist2[p, selected]
                
                # Uniqueness check
                if len(set(dvals)) != len(dvals):
                    valid = False
                    # If invalid, skip step 3
                else:
                    # Conflict check
                    for i, s in enumerate(selected):
                        if dvals[i] in point_dists[i]:
                            valid = False
                            break
            
            if valid:
                selected.append(p)
                
                # Calculate distances to existing points
                if selected:
                    # Distances to all previous points
                    # p is the new point (last in selected)
                    # Indices: 0 to len(selected)-2
                    new_dists = set()
                    # dvals was calculated if selected was not empty.
                    # But dvals includes distance to all selected BEFORE p was added.
                    # So dvals corresponds to selected[:-1]
                    # We need to add these to new_dists
                    if len(selected) > 1:
                        # dvals is the array of distances to previous points
                        new_dists = set(dvals)
                        
                        # Update point_dists for existing points
                        for i in range(len(selected) - 1):
                            point_dists[i].add(dvals[i])
                    else:
                        new_dists = set()
                
                point_dists.append(new_dists)
                
        # End of greedy attempt
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = [(idx // N, idx % N) for idx in selected]
            
            # Optional: Try to improve with local search?
            # Given time constraints, maybe just keep searching with random restarts.
            # But if we find a big one, we can try to add more?
            # The greedy is "first come, first served".
            # We could try a "repair" step:
            # 1. Identify points that were rejected.
            # 2. Try to swap?
            
            # For now, just record.
            
    return best_set
