import numpy as np
import itertools
import time

D = 11

def generate_candidates():
    """
    Generate all integer vectors with squared norm 4.
    Type: Weight 4, entries +/- 1. (5280 vectors)
    """
    candidates = []
    
    # Type 1: Weight 4, +/- 1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
                
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline, rng):
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    selected = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        chosen = candidates[rng.integers(len(candidates))]
        selected.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts
    return selected

def local_improve(selected, conflicts, deadline, rng):
    N = len(conflicts)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    
    # blocked_count[v] = number of selected vertices that conflict with v
    # Initially, for each selected u, increment blocked_count for all neighbors of u
    blocked_count = np.zeros(N, dtype=np.int32)
    for u in current_selected:
        neighbors = np.where(conflicts[u])[0]
        blocked_count[neighbors] += 1
    
    improved = True
    max_iterations = 500
    iteration = 0
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # 1. Try to add any vertex with blocked_count == 0
        # A vertex v is addable if ~selected_mask[v] and blocked_count[v] == 0
        addable = ~selected_mask & (blocked_count == 0)
        if np.any(addable):
            addable_indices = np.where(addable)[0]
            # Pick one randomly or with lowest degree?
            # Let's pick one that has the most neighbors (to block more others? No, we want to keep options open)
            # Actually, just pick one.
            v_in = addable_indices[rng.integers(len(addable_indices))]
            selected_mask[v_in] = True
            current_selected.append(v_in)
            
            # Update blocked_count
            neighbors = np.where(conflicts[v_in])[0]
            blocked_count[neighbors] += 1
            improved = True
            continue # Restart loop to check for more additions
            
        # 2. No addable vertices. Try 1-1 swap
        # For each u in selected, remove it, see if any vertex becomes addable.
        # If so, swap u -> v.
        # To be efficient, we can iterate through selected vertices.
        for idx in range(len(current_selected)):
            if time.time() > deadline:
                return current_selected
            
            u_out = current_selected[idx]
            
            # Remove u_out
            selected_mask[u_out] = False
            current_selected[idx] = -1 # Mark as removed temporarily
            
            neighbors = np.where(conflicts[u_out])[0]
            blocked_count[neighbors] -= 1
            
            # Check if any vertex is now addable
            # Note: u_out itself has blocked_count decreased by its neighbors? No, blocked_count is for OTHERS.
            # u_out is not selected, so we check ~selected_mask & (blocked_count == 0)
            # But we must exclude u_out from being added back immediately?
            # Actually, adding u_out back is a no-op.
            # We look for v != u_out such that v is addable.
            
            addable = ~selected_mask & (blocked_count == 0)
            # Exclude u_out if it's in the addable list (it shouldn't be if it conflicts with others? 
            # If u_out had no conflicts with others, it would be addable. 
            # If we add u_out back, we are back to start. 
            # We want a DIFFERENT vertex.
            if np.any(addable):
                addable_indices = np.where(addable)[0]
                # Filter out u_out
                if len(addable_indices) > 1 or addable_indices[0] != u_out:
                    # Pick a candidate
                    # Prefer candidates that are not u_out
                    candidates = addable_indices[addable_indices != u_out]
                    if len(candidates) > 0:
                        v_in = candidates[rng.integers(len(candidates))]
                        # Perform swap
                        selected_mask[v_in] = True
                        current_selected[idx] = v_in
                        
                        # Update blocked_count for v_in
                        v_neighbors = np.where(conflicts[v_in])[0]
                        blocked_count[v_neighbors] += 1
                        
                        improved = True
                        break # Restart loop
                    else:
                        # Only u_out was addable. Undo swap.
                        selected_mask[u_out] = True
                        current_selected[idx] = u_out
                        blocked_count[neighbors] += 1
                else:
                    # Only u_out was addable. Undo swap.
                    selected_mask[u_out] = True
                    current_selected[idx] = u_out
                    blocked_count[neighbors] += 1
            else:
                # No addable vertices. Undo swap.
                selected_mask[u_out] = True
                current_selected[idx] = u_out
                blocked_count[neighbors] += 1
                
            if improved:
                break

    return current_selected

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    vectors = generate_candidates()
    N = len(vectors)
    
    dots = vectors @ vectors.T
    # To ensure distance > max norm (strict separation to avoid "overlap" errors),
    # we require d^2 > 4.
    # d^2 = 8 - 2*dot.
    # 8 - 2*dot > 4 => 2*dot < 4 => dot < 2.
    # Since dot is integer, dot <= 1.
    # So conflict if dot >= 2.
    conflicts = (dots >= 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    mis_deadline = deadline - 40
    while time.time() < mis_deadline:
        current_set = greedy_mis(conflicts, mis_deadline, rng)
        if len(current_set) > len(best_set):
            best_set = current_set
        # Early stop if we hit a high bound
        if len(best_set) >= 500:
            break
            
    if len(best_set) > 0 and time.time() < deadline - 5:
        improved_set = local_improve(list(best_set), conflicts, deadline - 2, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    if not best_set:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    return vectors[best_set].astype(np.int64)
