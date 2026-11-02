import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build compatibility matrix: compat[i,j] = True if M[i,j] <= 2
    # Note: M[i,i] = 4, so we must exclude self or handle it.
    # We want i != j.
    compat = (M <= 2)
    np.fill_diagonal(compat, False)
    
    # Convert to boolean array for fast bitwise operations
    # Note: compat is already boolean from (M <= 2)
    
    # Helper function to extend a set greedily
    def extend_set(selected_indices, rng):
        selected = set(selected_indices)
        available = np.ones(N, dtype=bool)
        available[list(selected)] = False
        
        # Precompute intersection of compat for selected?
        # It's expensive to recompute from scratch every time.
        # But N is ~5300, N^2 is ~28M. Bitwise AND of 5300 length arrays is fast.
        
        # Current available is intersection of compat[v] for v in selected
        # Let's maintain it incrementally if possible, or just recompute if set is small.
        # For local search, we might do many extensions.
        
        # Strategy: 
        # 1. Compute 'addable' mask.
        # 2. If any, pick one (random or best degree) and add to selected, update available.
        
        while True:
            # Compute addable
            # Start with all True, then AND with compat[v] for each v in selected
            # This is O(|S| * N / 64) operations.
            # If |S| is large (e.g., 500), this is 500 * 5300 / 64 ~ 40k ops. Very fast.
            
            addable = np.ones(N, dtype=bool)
            for v in selected:
                addable &= compat[v]
            
            # Exclude already selected
            for v in selected:
                addable[v] = False
                
            if not np.any(addable):
                break
            
            # Pick a candidate
            # Option 1: Random
            # Option 2: Highest degree in the induced subgraph of addable?
            # Let's stick to random for simplicity and speed, or pick the first one.
            # Random is good for exploration.
            candidates_to_add = np.where(addable)[0]
            if len(candidates_to_add) == 0:
                break
            # Pick one at random
            idx = candidates_to_add[rng.integers(len(candidates_to_add))]
            selected.add(idx)
            
        return selected

    # Initial Greedy to get a good starting point
    # Use the spectral order or degree order
    deg = (~compat).sum(axis=1)
    
    best_set = None
    best_count = 0
    rng = np.random.default_rng(42)
    
    # Run a few greedy initializations
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        if trial == 0:
            order = np.argsort(deg, kind='stable')
        else:
            # Random permutation
            order = np.random.permutation(N)
            
        selected = set()
        # Greedy add
        # We can use the extend_set logic but it's iterative.
        # Let's do a simple greedy pass first.
        available = np.ones(N, dtype=bool)
        for idx in order:
            if available[idx]:
                selected.add(idx)
                available &= compat[idx]
                if not np.any(available):
                    break
        
        # Now extend it
        extended = extend_set(list(selected), rng)
        if len(extended) > best_count:
            best_count = len(extended)
            best_set = extended.copy()

    if best_set is None:
        return candidates[:22].astype(np.int64)

    # Local Search Phase
    # Current best set
    current_set = best_set.copy()
    current_size = len(current_set)
    
    # Precompute degrees for heuristic
    # We will try:
    # 1. Random Remove & Extend
    # 2. Swap (Remove 1, Add 1+ to keep size or increase)
    
    # To speed up "Extend", we can cache the "addable" mask for the current set.
    # But it changes when we remove/add.
    
    # Let's implement a loop that runs until deadline
    while time.time() < deadline - 5:
        # Try to extend current_set (should be maximal already from previous step, 
        # but if we did a swap, it might not be)
        extended = extend_set(list(current_set), rng)
        if len(extended) > current_size:
            current_set = extended
            current_size = len(current_set)
            if current_size > best_count:
                best_count = current_size
                best_set = current_set.copy()
        
        # If we have time, try a move
        if time.time() > deadline - 5:
            break
            
        # Move: Remove a random element, then extend
        # Or: Remove k random elements, extend, if size increased, keep it.
        
        # Let's try removing 1 element
        if len(current_set) > 0:
            # Pick a random element to remove
            # Bias towards elements that are "central" (high degree) to free up space?
            # Or random.
            elements = list(current_set)
            remove_idx = elements[rng.integers(len(elements))]
            
            new_set = current_set.copy()
            new_set.remove(remove_idx)
            
            # Extend the new set
            new_extended = extend_set(list(new_set), rng)
            
            if len(new_extended) > current_size:
                current_set = new_extended
                current_size = len(current_set)
                if current_size > best_count:
                    best_count = current_size
                    best_set = current_set.copy()
            # else: revert (current_set unchanged)
            
            # Optional: Try removing 2 elements
            # This is more expensive but can jump out of local optima
            # if time.time() < deadline - 10 and len(current_set) > 1:
            #    ...

    return candidates[list(best_set)].astype(np.int64)
