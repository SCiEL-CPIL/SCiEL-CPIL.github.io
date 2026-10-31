import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: All vectors with exactly two non-zero entries (+/-1).
    # Norm^2 = 2. Max norm = sqrt(2).
    # Min distance between distinct such vectors is sqrt(2) (e.g., e1+e2 and e1+e3 -> diff e2-e3, norm sqrt(2)).
    # So all 220 are valid.
    candidates_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_2.append(v)
    
    baseline = np.array(candidates_2, dtype=np.int64)
    best_set = baseline
    best_size = len(baseline)
    
    # Try to find a larger set from Norm^2=4 vectors.
    # If we use only Norm^2=4, max norm = 2. Need dist^2 >= 4.
    
    candidates_4 = []
    # Type A: One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates_4.append(v)
            
    # Type B: Four entries +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)
    n4 = len(candidates_4)
    
    if time.time() > deadline - 20:
        return best_set
    
    # Compute conflict matrix for candidates_4
    # dist^2 = n_i + n_j - 2 dot
    # All norms are 4. dist^2 = 8 - 2 dot.
    # Conflict if dist^2 < 4 => 8 - 2 dot < 4 => 2 dot > 4 => dot > 2.
    # dot is integer. So conflict if dot >= 3.
    
    dots = candidates_4 @ candidates_4.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, False)
    
    # Greedy Independent Set with Random Restarts
    # We want a large set of nodes with no conflicts.
    
    np.random.seed(42)
    
    best_4_set = []
    
    # Number of restarts limited by time
    max_restarts = 100
    restarts_done = 0
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 5:
            break
        restarts_done += 1
        
        remaining = np.ones(n4, dtype=bool)
        # Initial degrees
        degrees = conflict.sum(axis=1).astype(np.int32)
        
        selected = []
        
        # Greedy: pick node with min degree, remove it and its neighbors
        # To speed up, we can just pick random node from remaining if time is tight,
        # but min-degree is better.
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Get degrees of remaining
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            # Pick one of the min-degree nodes
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and neighbors
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees
                # deg_new[r] = deg_old[r] - count(neighbors of r in actual_remove)
                deg_decrease = conflict[remaining][:, actual_remove].sum(axis=1)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
            
            # Time check
            if time.time() > deadline - 5:
                break
                
        if len(selected) > len(best_4_set):
            best_4_set = selected
            
    if len(best_4_set) > best_size:
        best_set = candidates_4[best_4_set]
        best_size = len(best_4_set)
        
    # Try to improve with local search (swap-based)
    # For each selected vector, try swapping with a non-selected vector if it reduces conflicts
    # This is a refinement step
    
    if best_size > 0 and len(best_4_set) > 0:
        # Convert best_4_set to a boolean mask
        mask = np.zeros(n4, dtype=bool)
        mask[best_4_set] = True
        
        # Compute degree of each node in the selected set
        # A node is "valid" if it has no conflicts with any selected node
        # We want to swap a selected node with a non-selected node that has fewer conflicts
        
        # For efficiency, we can compute the conflict count for each non-selected node
        # with the current selected set
        
        # But this might be too slow. Let's stick with the greedy approach for now.
        
        # Alternative: Try to add more vectors if possible
        # Check if there are any non-selected vectors that can be added without conflict
        # This is a greedy addition step
        
        # Compute the set of conflicts with the current selected set
        # conflict_with_selected[i] = number of selected nodes that conflict with i
        conflict_with_selected = (conflict[:, mask].sum(axis=1))
        
        # Find non-selected nodes with 0 conflicts
        addable = (~mask) & (conflict_with_selected == 0)
        
        if np.any(addable):
            # Add all addable nodes
            new_indices = np.where(addable)[0]
            for idx in new_indices:
                mask[idx] = True
            
            # Update best_4_set
            best_4_set = np.where(mask)[0].tolist()
            best_size = len(best_4_set)
            best_set = candidates_4[best_4_set]
            
            # Repeat until no more can be added
            while True:
                conflict_with_selected = (conflict[:, mask].sum(axis=1))
                addable = (~mask) & (conflict_with_selected == 0)
                
                if not np.any(addable):
                    break
                
                new_indices = np.where(addable)[0]
                for idx in new_indices:
                    mask[idx] = True
                
                best_4_set = np.where(mask)[0].tolist()
                best_size = len(best_4_set)
                best_set = candidates_4[best_4_set]
                
                if time.time() > deadline - 5:
                    break
    
    return best_set
