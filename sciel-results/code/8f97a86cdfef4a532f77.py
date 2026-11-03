import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Generate all weight-4 vectors (four ±1 entries)
    # Norm squared is 4. Constraint: dot product <= 2
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute dot product matrix and conflict matrix
    dots = candidates @ candidates.T
    # Conflict if dot > 2. Diagonal is 4, but we ignore self.
    conflict = (dots > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Build a helper for fast conflict counting
    # conflict[i] gives a uint8 array of conflicts for candidate i
    
    def greedy_construct(order):
        in_set = np.zeros(n_cand, dtype=bool)
        selected = []
        
        for i in order:
            if time.time() > deadline - 30:
                break
            if in_set[i]:
                continue
            # Check if i conflicts with any selected node
            if np.sum(conflict[i][in_set]) == 0:
                in_set[i] = True
                selected.append(i)
                
        return in_set, selected

    def local_search(in_set, selected):
        # Phase 1: Fill in any nodes that don't conflict
        improved = True
        while improved:
            if time.time() > deadline - 20:
                return in_set, selected
            improved = False
            # Vectorized check: for all unselected, count conflicts
            unselected_mask = ~in_set
            if not np.any(unselected_mask):
                break
            # Count conflicts for each unselected node with current set
            conflict_counts = conflict[unselected_mask] @ in_set
            free_indices = np.where(conflict_counts == 0)[0]
            if len(free_indices) > 0:
                # Add the first free index (or all? Adding all at once is safe if they don't conflict with each other? 
                # No, two free nodes might conflict with each other. So we must add one by one or check batch compatibility.
                # For speed, let's add one and restart. Or add a batch that is mutually non-conflicting?
                # Greedy: pick one, add it, restart loop.
                idx = free_indices[0]
                actual_idx = np.where(unselected_mask)[0][idx]
                in_set[actual_idx] = True
                selected.append(actual_idx)
                improved = True
        
        # Phase 2: Swap moves
        # Try to remove one node and add more.
        # To be efficient, we only try a limited number of swaps.
        max_swaps = 100
        for _ in range(max_swaps):
            if time.time() > deadline - 10:
                break
            if not selected:
                break
            
            # Pick a random node to remove
            u_idx = random.choice(selected)
            u_pos = selected.index(u_idx)
            
            # Temporarily remove
            in_set[u_idx] = False
            selected.pop(u_pos)
            
            # Try to add as many as possible
            added = 0
            # We can do a quick greedy fill
            # But we need to be careful not to exceed time.
            # Let's do a limited fill
            for _ in range(10):  # Limit inner loop
                if time.time() > deadline - 10:
                    break
                unselected_mask = ~in_set
                if not np.any(unselected_mask):
                    break
                conflict_counts = conflict[unselected_mask] @ in_set
                free_indices = np.where(conflict_counts == 0)[0]
                if len(free_indices) == 0:
                    break
                idx = free_indices[0]
                actual_idx = np.where(unselected_mask)[0][idx]
                in_set[actual_idx] = True
                selected.append(actual_idx)
                added += 1
            
            # If we didn't improve, we could revert, but greedy fill is monotonic in terms of "trying".
            # However, removing u might have allowed adding nodes that block better future moves.
            # Given time, we accept the result.
            
        return in_set, selected

    best_in_set = None
    best_k = 0

    # Multiple trials with random orderings
    # Number of trials depends on time remaining
    # Let's do at least 5, up to 50 if fast
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        order_rand = np.random.permutation(n_cand)
        in_set, selected = greedy_construct(order_rand)
        in_set, selected = local_search(in_set, selected)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_in_set = in_set.copy()
            # Optional: update random seed or just continue

    if best_in_set is None:
        # Fallback
        in_set = np.zeros(n_cand, dtype=bool)
        selected = []
        for i in range(n_cand):
            if np.sum(conflict[i][in_set]) == 0:
                in_set[i] = True
                selected.append(i)
        best_in_set = in_set
        best_k = len(selected)

    result = candidates[best_in_set]
    return result.astype(np.int64)
