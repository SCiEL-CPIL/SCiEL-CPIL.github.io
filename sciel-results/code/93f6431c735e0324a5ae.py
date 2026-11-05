import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances using numpy for speed
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    best_set = []
    best_size = 0
    
    # Heuristic: prioritize points that are "spread out" or have rare distance patterns?
    # For now, random restarts with greedy addition.
    # To improve efficiency, we can use a "conflict graph" style approach or just optimized greedy.
    
    # Optimization idea: 
    # Instead of iterating all 4096 points for each attempt, we can pre-shuffle.
    # The inner loop is the bottleneck. 
    # Let's try to make the inner loop as fast as possible.
    
    # We store the current set indices in a list `si`.
    # We store for each point in the set, the set of distances to other points in the set.
    # `ud[i]` is the set of squared distances from point si[i] to all other points in the set.
    # When adding a new point p, we check:
    # 1. All distances from p to existing points are distinct.
    # 2. For each existing point q, the distance d(p,q) is not already in ud[q].
    
    # We can vectorize the distance extraction if we keep si as a numpy array?
    # But si changes size. A list is fine for K < 100.
    
    # Let's try a different heuristic: 
    # Sort points by some criteria before greedy?
    # Or just random order.
    
    # Another idea: Local Search.
    # Start with a random valid set, try to swap or add points.
    # But greedy is simple and effective for this kind of packing problem.
    
    # Let's optimize the greedy loop.
    
    for attempt in range(5000): # Increased attempts, rely on time check
        if time.time() > deadline:
            break
        
        # Random permutation of all points
        # Using random.shuffle on a list of 4096 ints is fast
        perm = list(range(TOTAL))
        random.shuffle(perm)
        
        si = []
        ud = [] # list of sets
        
        for p_idx in perm:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            # Check distinct distances from p to all in si
            # We need the distances for the next check too, so compute them once.
            # d_vec = dist2[p_idx, si] # This creates a numpy array view/slice
            # But si is a list. dist2[p_idx, si] works if si is a list of indices?
            # Yes, numpy supports fancy indexing with lists.
            # However, creating the array might have overhead if K is small.
            # Let's try fancy indexing.
            
            # If K is small, manual loop might be faster due to numpy overhead.
            # Let's stick to manual loop but optimized.
            
            valid = True
            dists = []
            for q_idx in si:
                d = dist2[p_idx, q_idx]
                dists.append(d)
            
            # Check distinctness
            if len(set(dists)) != len(dists):
                continue
            
            # Check against existing used distances
            for i in range(len(si)):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if valid:
                si.append(p_idx)
                # New set for the new point
                new_ud = set(dists)
                ud.append(new_ud)
                
                # Update existing points' used distance sets
                for i in range(len(si) - 1):
                    ud[i].add(dists[i])
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
            
            # If we get a very large set, we can try to refine it with local search
            # or just keep searching.
            
    return best_set
