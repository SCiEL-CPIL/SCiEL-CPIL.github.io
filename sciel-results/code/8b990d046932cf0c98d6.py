"""Solve kissing number in dimension 11 via simulated annealing on conflict graph."""
import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # We need integer vectors in R^11 with:
    # - All nonzero
    # - min pairwise distance >= max norm
    # 
    # If all vectors have the same norm r, then we need ||a-b|| >= r for all pairs.
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b = 2r^2 - 2a.b >= r^2
    # => a.b <= r^2/2
    #
    # For integer vectors, let's use vectors with squared norm 4 (norm = 2).
    # Then we need a.b <= 2 for all pairs.
    #
    # Integer vectors with squared norm 4 in R^11:
    # Type A: one entry is ±2, rest 0. (11 * 2 = 22 vectors)
    # Type B: four entries are ±1, rest 0. (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    # Total: 5302 candidates

    # Generate all candidates
    candidates = []
    
    # Type A: one entry ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    # Type B: four entries ±1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)
    
    cands = np.array(candidates, dtype=np.float64)
    n_cands = len(cands)
    
    # Precompute the conflict graph: conflict[i,j] = True if dot(cands[i], cands[j]) > 2
    # This is an n_cands x n_cands matrix, but n_cands = 5302 so it's ~28M entries.
    # We can compute it in blocks to save memory.
    
    # Actually, let's use a different approach. We'll use simulated annealing.
    # State: a subset S of candidates (represented as a boolean mask or list of indices)
    # Objective: maximize |S| subject to no conflicts within S.
    # 
    # Simulated annealing moves:
    # 1. Add a random candidate to S (if no conflict, accept; if conflict, accept with prob exp(-delta/T))
    # 2. Remove a random candidate from S
    # 3. Swap: remove one, add another
    #
    # Energy function: E = number of conflicts within S - lambda * |S|
    # We want to minimize E, where lambda is a large positive number.
    
    # For efficiency, let's precompute dot products in a smarter way.
    # Since all vectors have norm^2 = 4, dot(a,b) = (4 - ||a-b||^2)/2
    # Conflict if dot > 2, i.e., 4 - ||a-b||^2 > 4, i.e., ||a-b||^2 < 0. 
    # Wait, that's wrong. Let me recompute.
    # dot(a,b) = (||a||^2 + ||b||^2 - ||a-b||^2)/2 = (4 + 4 - ||a-b||^2)/2 = (8 - ||a-b||^2)/2
    # Conflict if dot > 2: (8 - ||a-b||^2)/2 > 2 => 8 - ||a-b||^2 > 4 => ||a-b||^2 < 4
    # So conflict if ||a-b||^2 < 4, i.e., distance < 2.
    # No conflict if ||a-b||^2 >= 4, i.e., distance >= 2.
    
    # Let's precompute the conflict matrix using block computation
    # For n_cands = 5302, the full matrix is 5302^2 = 28M entries, which is ~224MB in float64.
    # That's manageable.
    
    # Compute dot product matrix
    # dots = cands @ cands.T  # This would be 5302 x 5302
    # But that's expensive. Let's do it in blocks.
    
    # Actually, let's just compute it directly since NumPy is fast.
    dots = cands @ cands.T
    # Conflict if dot > 2
    conflict = (dots > 2 + 1e-9).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    del dots
    
    # Now use simulated annealing
    # State: boolean mask of selected candidates
    # We'll represent state as a set of indices for efficient conflict checking
    
    rng = np.random.RandomState(42)
    
    # Initial state: greedy selection
    selected = set()
    order = rng.permutation(n_cands)
    for idx in order:
        if time.time() > deadline - 30:
            break
        # Check if idx conflicts with any selected
        # conflict[idx, list(selected)] 
        if len(selected) == 0:
            selected.add(idx)
        else:
            sel_arr = np.array(list(selected))
            if not np.any(conflict[idx, sel_arr]):
                selected.add(idx)
    
    best_selected = set(selected)
    best_size = len(selected)
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.1
    n_iterations = 200000
    lambda_weight = 10.0  # weight for size in energy
    
    # Energy: E = n_conflicts - lambda_weight * |S|
    # We want to minimize E
    
    def get_energy(sel_set):
        if len(sel_set) == 0:
            return 0.0
        sel_arr = np.array(list(sel_set))
        n_conf = np.sum(conflict[sel_arr, sel_arr]) // 2
        return n_conf - lambda_weight * len(sel_set)
    
    current_energy = get_energy(selected)
    
    # Precompute for speed: for each candidate, which others does it conflict with?
    # conflict is already a matrix, so conflict[i] gives the conflict mask for i.
    
    for it in range(n_iterations):
        if time.time() > deadline - 10:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        # Random move
        move_type = rng.randint(3)
        
        if move_type == 0 and len(selected) < n_cands:
            # Add a random candidate
            idx = rng.randint(n_cands)
            if idx in selected:
                continue
            # Check conflicts with current selected
            if len(selected) > 0:
                sel_arr = np.array(list(selected))
                n_new_conf = np.sum(conflict[idx, sel_arr])
            else:
                n_new_conf = 0
            
            delta_E = n_new_conf - lambda_weight  # adding one element
            
        elif move_type == 1 and len(selected) > 0:
            # Remove a random candidate
            sel_list = list(selected)
            idx = sel_list[rng.randint(len(sel_list))]
            # Count conflicts that idx has with other selected
            if len(selected) > 1:
                other_sel = np.array([x for x in selected if x != idx])
                n_removed_conf = np.sum(conflict[idx, other_sel])
            else:
                n_removed_conf = 0
            
            delta_E = -n_removed_conf + lambda_weight  # removing one element
            
        elif move_type == 2 and len(selected) > 0:
            # Swap: remove one, add another
            sel_list = list(selected)
            idx_remove = sel_list[rng.randint(len(sel_list))]
            idx_add = rng.randint(n_cands)
            if idx_add in selected:
                continue
            
            # Remove idx_remove, add idx_add
            # New conflicts: conflicts of idx_add with (selected - {idx_remove})
            if len(selected) > 1:
                other_sel = np.array([x for x in selected if x != idx_remove])
                new_conf = np.sum(conflict[idx_add, other_sel])
                # Old conflicts removed: conflicts of idx_remove with other_sel
                old_conf = np.sum(conflict[idx_remove, other_sel])
            else:
                new_conf = 0
                old_conf = 0
            
            delta_E = new_conf - old_conf  # size stays the same
        
        else:
            continue
        
        # Acceptance
        if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
            # Apply move
            if move_type == 0:
                selected.add(idx)
            elif move_type == 1:
                selected.discard(idx)
            elif move_type == 2:
                selected.discard(idx_remove)
                selected.add(idx_add)
            
            current_energy += delta_E
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
    
    # Final greedy improvement pass
    # Try to add any remaining candidates that don't conflict
    for idx in range(n_cands):
        if time.time() > deadline - 5:
            break
        if idx in best_selected:
            continue
        if len(best_selected) > 0:
            sel_arr = np.array(list(best_selected))
            if not np.any(conflict[idx, sel_arr]):
                best_selected.add(idx)
    
    # Convert to array
    if len(best_selected) > 0:
        result = cands[list(best_selected)].astype(np.int64)
    else:
        result = cands[:22].astype(np.int64)
    
    # Verify
    if len(result) > 1:
        norms_sq = np.sum(result * result, axis=1)
        max_norm = np.max(norms_sq)
        n = len(result)
        if n <= 2000:
            valid = True
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, n):
                    diff = result[i] - result[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < max_norm - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                # Fall back to safe set
                return cands[:22].astype(np.int64)
    
    return result
