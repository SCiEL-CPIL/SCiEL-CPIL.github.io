import numpy as np
import itertools
import time

D = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate all integer vectors with norm squared 3
    # These are vectors with three non-zero entries, each +/- 1
    candidates = []
    # Indices for 3 non-zero positions
    indices = list(itertools.combinations(range(D), 3))
    # Signs
    signs = list(itertools.product([1, -1], repeat=3))
    
    for idx in indices:
        for s in signs:
            v = np.zeros(D, dtype=np.int64)
            v[idx[0]] = s[0]
            v[idx[1]] = s[1]
            v[idx[2]] = s[2]
            candidates.append(v)
            
    candidates = np.array(candidates)
    n = len(candidates)
    
    # Compute dot products
    # Dot product matrix
    # v_i . v_j
    dot_products = candidates @ candidates.T
    
    # Conflict if dot product >= 2
    # We want independent set where dot product <= 1
    conflict = (dot_products >= 2)
    
    best_set = []
    best_size = 0
    
    # Greedy MIS with random restarts
    # Number of restarts depends on time
    # Each restart is O(n^2) roughly, but n=1320 is small.
    # 1320^2 = 1.7M ops. Very fast.
    # We can do thousands of restarts.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        remaining = np.ones(n, dtype=bool)
        selected = []
        
        # Shuffle nodes for randomization
        order = np.random.permutation(n)
        
        # Greedy selection
        # We pick nodes in random order, but we can improve by picking min-degree
        # For simplicity and speed, let's use a simple greedy with random order first
        # Or better: min-degree heuristic
        
        # Min-degree heuristic is better for MIS
        # Degree of a node is number of conflicts among remaining nodes
        
        # We will use a simple iterative removal
        # To speed up, we can use vectorized operations
        
        current_remaining = np.ones(n, dtype=bool)
        # Initialize degrees
        # Degree of i is sum of conflict[i, j] for j in remaining
        # We can update degrees as we remove nodes
        
        # Actually, for n=1320, we can just loop
        selected_indices = []
        
        # Copy conflict matrix for mutation? No, just use boolean mask
        # Let's use a list of remaining nodes
        remaining_list = list(range(n))
        
        # To make it faster, we can precompute adjacency lists
        # But vectorized numpy is good.
        
        # Let's try a simple greedy:
        # At each step, pick a random node from remaining, add to set, remove it and neighbors.
        # This is O(k * n^2) worst case.
        
        # Better: 
        # 1. Compute degrees
        # 2. Pick min degree
        # 3. Remove
        
        # Let's implement a fast min-degree greedy
        
        active = np.ones(n, dtype=bool)
        # Degrees
        # deg[i] = sum(conflict[i, active])
        # We can compute this vectorized
        
        while active.any():
            # Compute degrees for active nodes
            # conflict is n x n boolean
            # deg = conflict @ active (as int)
            deg = (conflict @ active.astype(np.int8)).astype(np.int32)
            
            # Find active node with min degree
            # Set degrees of inactive nodes to inf
            deg[~active] = n + 1
            
            # Pick min
            min_idx = np.argmin(deg)
            
            # Add to set
            selected_indices.append(min_idx)
            
            # Remove min_idx and its neighbors
            # Neighbors are conflict[min_idx, :] & active
            # Remove them from active
            active[min_idx] = False
            active[conflict[min_idx] & active] = False
            
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_set = selected_indices
            
    # Convert best_set to array of vectors
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set]
