import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s margin
    
    # Generate weight-2 vectors (Seed)
    vecs_w2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs_w2.append(v)
    vecs_w2 = np.array(vecs_w2)
    k_w2 = len(vecs_w2) # 220
    
    # Generate weight-3 vectors
    vecs_w3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vecs_w3.append(v)
    vecs_w3 = np.array(vecs_w3)
    
    # We need subset of vecs_w3 with min sq dist >= 3
    # Conflict if sq dist < 3 (i.e. 1 or 2)
    # Sq dist 1 is impossible for weight 3 vectors (min diff 2)
    # So conflict if sq dist == 2
    
    # Greedy Independent Set
    # 1. Calculate degrees
    # 2. Sort by degree
    # 3. Pick, remove neighbors
    
    n = len(vecs_w3)
    if n == 0:
        return vecs_w2
        
    # Efficient distance check?
    # n=1320. n^2 = 1.7M.
    # Dot products: vecs_w3 @ vecs_w3.T
    # Sq dist = 3 + 3 - 2 dot = 6 - 2 dot
    # Conflict if 6 - 2 dot < 3 => 2 dot > 3 => dot > 1.5 => dot >= 2
    
    # Calculate dot products
    dots = vecs_w3 @ vecs_w3.T
    # Conflict matrix
    # Diagonal is 3 (dist 0). We ignore self.
    # Conflict if dot >= 2
    conflict_mask = (dots >= 2)
    np.fill_diagonal(conflict_mask, False)
    
    # Degree of each vertex
    degrees = np.sum(conflict_mask, axis=1)
    
    # Greedy: pick vertex with min degree, remove it and its neighbors
    available = np.ones(n, dtype=bool)
    selected_indices = []
    
    # To speed up, we can use a priority queue or just simple loop if n is small
    # n=1320 is small enough for O(n^2) greedy with numpy ops
    
    while np.sum(available) > 0:
        # Find min degree among available
        # Set unavailable degrees to inf
        degs_avail = np.where(available, degrees, np.inf)
        min_idx = np.argmin(degs_avail)
        
        if not available[min_idx]:
            break
            
        selected_indices.append(min_idx)
        
        # Remove min_idx and its neighbors
        available[min_idx] = False
        neighbors = np.where(conflict_mask[min_idx] & available)[0]
        available[neighbors] = False
        
        # Update degrees? 
        # For greedy, we don't strictly need to update degrees if we just pick min of current available.
        # But degrees change as neighbors are removed.
        # Recalculating degrees is O(n^2).
        # Total O(n^3) worst case? 1320^3 is too slow.
        
        # Optimization: Just pick random available? Or min degree without update?
        # Min degree without update is faster.
        
        if time.time() > deadline:
            break

    if len(selected_indices) > k_w2:
        return vecs_w3[selected_indices]
    else:
        return vecs_w2
