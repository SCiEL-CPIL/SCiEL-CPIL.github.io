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
    deadline = start_time + 115  # Safety margin (150s limit, leave 35s buffer)
    
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
    
    # Greedy Set-Packing with "Future Conflict Radius" heuristic
    # Strategy: Pick nodes that eliminate the fewest remaining compatible partners.
    # This preserves search space flexibility.
    
    np.random.seed(42)
    
    best_4_set = []
    
    # Number of restarts limited by time
    max_restarts = 50
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        remaining = np.ones(n4, dtype=bool)
        selected = []
        
        # We maintain a count of "compatible partners" for each remaining node.
        # compatible_count[i] = number of j in remaining such that not conflict[i, j]
        # Initially, compatible_count[i] = n4 - 1 - degree[i]
        # But we need to update this as we remove nodes.
        
        # For efficiency, let's use a heuristic score.
        # score[i] = number of remaining nodes that i is compatible with.
        # We want to pick the node with the MINIMUM score (eliminates fewest options).
        
        # Initial scores:
        # compatible_count[i] = (n4 - 1) - conflict[i].sum()
        scores = (n4 - 1) - conflict.sum(axis=1).astype(np.int32)
        
        while remaining.any():
            if time.time() > deadline - 10:
                break
            
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Get scores of remaining
            rem_scores = scores[rem_indices]
            min_score = np.min(rem_scores)
            
            # Pick one of the min-score nodes
            min_mask = rem_scores == min_score
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors from remaining
            # Neighbors are those with conflict[chosen_global, j] == True
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update scores for remaining nodes
                # For each remaining node r, score[r] decreases by the number of removed nodes
                # that were compatible with r.
                # compatible(r, u) = not conflict[r, u]
                # So decrease = count(u in actual_remove such that not conflict[r, u])
                
                # Let's compute this efficiently.
                # conflict[remaining][:, actual_remove] gives a matrix where entry (r, u) is 1 if conflict.
                # We want (1 - conflict) for each pair.
                # num_compatible_removed = len(actual_remove) - conflict[remaining][:, actual_remove].sum(axis=1)
                
                num_removed = len(actual_remove)
                conflict_count = conflict[remaining][:, actual_remove].sum(axis=1)
                compatible_removed = num_removed - conflict_count
                
                scores[remaining] -= compatible_removed
                
            remaining[actual_remove] = False
            
            # Time check
            if time.time() > deadline - 10:
                break
        
        if len(selected) > len(best_4_set):
            best_4_set = selected
            
    if len(best_4_set) > best_size:
        best_set = candidates_4[best_4_set]
        best_size = len(best_4_set)
        
    # Try to improve with local search (add-based)
    # For each non-selected vector, try adding it if it has no conflicts with the current set
    
    if best_size > 0 and len(best_4_set) > 0:
        # Convert best_4_set to a boolean mask
        mask = np.zeros(n4, dtype=bool)
        mask[best_4_set] = True
        
        # Repeat until no more can be added
        while True:
            # Compute the set of conflicts with the current selected set
            # conflict_with_selected[i] = number of selected nodes that conflict with i
            conflict_with_selected = (conflict[:, mask].sum(axis=1))
            
            # Find non-selected nodes with 0 conflicts
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
