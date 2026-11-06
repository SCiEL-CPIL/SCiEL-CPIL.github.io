import random
import time
import math
import numpy as np

N = 64

# Pre-compute all grid points as a numpy array for vectorized operations
ALL_POINTS = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin for safety
    
    best_set = []
    best_len = 0
    
    # We will use multiple restarts with different random orderings.
    # Within each restart, we use a greedy algorithm.
    # To speed up the greedy step, we can try to be smarter about candidate selection
    # or just rely on the fact that N=64 is small enough that even O(N^3) per restart
    # might be fast if we limit the number of points added or optimize the inner loop.
    
    # However, the current program is pure Python loops.
    # Let's try to optimize the inner check using numpy vectorization.
    
    num_trials = 0
    
    while time.time() < deadline:
        num_trials += 1
        
        # Random permutation of indices
        indices = np.random.permutation(len(ALL_POINTS))
        
        S = []  # List of selected points (as tuples or indices)
        S_coords = [] # List of (x,y) for the selected points
        
        # For fast checking, we maintain:
        # 1. The list of selected points
        # 2. A set of all squared distances between pairs in S
        
        dist_set = set()
        
        # We iterate through the random order
        for idx in indices:
            p = ALL_POINTS[idx]
            px, py = p[0], p[1]
            
            m = len(S_coords)
            
            if m < 2:
                S_coords.append(p)
                if m == 1:
                    d = (px - S_coords[0][0])**2 + (py - S_coords[0][1])**2
                    dist_set.add(d)
                continue
            
            # Calculate squared distances from p to all points in S_coords
            # S_coords is a list of np arrays. Let's convert to numpy array for batch calc?
            # Or just loop, since m is likely small (maybe < 10-20).
            
            dists = []
            conflict = False
            for s in S_coords:
                dx = px - s[0]
                dy = py - s[1]
                d = dx*dx + dy*dy
                dists.append(d)
                if d in dist_set:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Check if all dists are distinct
            if len(set(dists)) != m:
                continue
            
            # Valid point
            dist_set.update(dists)
            S_coords.append(p)
            
            # Optional: If we hit a time limit inside the loop, break?
            # The outer while loop handles the restarts.
            # If we finish the loop, we have a maximal set for this ordering.
            
            # Check time periodically to avoid long single trials?
            # If a single trial takes too long, we might miss the deadline.
            # But N=64, so max 4096 points. Inner loop is O(m).
            # Worst case m=4096, inner loop 4096 ops. 4096*4096 ~ 16M ops.
            # In Python, this might take ~1-2 seconds per trial if m is large.
            # But m won't be large. The set will be small.
            # Let's assume m < 50. Then inner loop is fast.
            
            # We can add a time check here to break out of the inner loop if we're running out of time
            # but that would result in a non-maximal set. Better to finish the trial if possible.
            # Given the deadline is 280s, and we expect many trials, let's just let it run.
            # If a trial takes > 5s, we might want to break.
            # Let's add a check every 1000 points? No, m is small.
            
        # End of trial
        if len(S_coords) > best_len:
            best_len = len(S_coords)
            best_set = S_coords[:]
            
            # If we find a very good set, we might want to stop?
            # Or just keep looking for better.
            # Theoretical max is unknown, but likely < 100.
            # Let's keep looking.

    return [(int(p[0]), int(p[1])) for p in best_set]
