import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering and random tie-breaking.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive nodes
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def local_improve_fast(selected, vectors, conflicts, deadline, rng):
    """
    Fast local improvement using incremental compatibility counting.
    
    For each unselected vertex u, track comp_count[u] = number of selected vertices
    that conflict with u. A vertex is a valid swap-in candidate iff comp_count[u] == 0.
    When we swap out s and in u:
      - For each other selected v: if conflicts[u,v] then comp_count[v] changes... 
      Actually we track: for unselected x, comp_count[x] = sum over selected s of conflicts[x,s].
      When we remove s_out and add u_in:
        comp_count[x] -= conflicts[x, s_out] for all unselected x
        comp_count[x] += conflicts[x, u_in] for all unselected x
    """
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    k = len(current_selected)
    
    # Precompute comp_count for all unselected vertices
    # comp_count[i] = number of selected vertices that conflict with i
    if k > 0:
        comp_count = conflicts[:, selected_mask].sum(axis=1).astype(np.int32)
    else:
        comp_count = np.zeros(N, dtype=np.int32)
    
    # Mask of unselected
    unselected_mask = ~selected_mask
    
    improved = True
    max_iterations = 5000
    iteration = 0
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Find all valid swap-in candidates: unselected with comp_count == 0
        valid_in = unselected_mask & (comp_count == 0)
        
        if not np.any(valid_in):
            break
        
        # Try each selected vertex for potential swap-out
        # Shuffle to avoid bias
        order = list(range(k))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                return current_selected
            
            s_out = current_selected[idx]
            
            # Check: if we remove s_out, which unselected vertices become valid?
            # A vertex x (currently unselected) is valid after removing s_out if:
            #   comp_count[x] - conflicts[x, s_out] == 0
            # i.e., comp_count[x] == conflicts[x, s_out] (which is 0 or 1)
            
            # Find unselected x where comp_count[x] == 1 and conflicts[x, s_out] == 1
            # OR comp_count[x] == 0 (already valid, but those are already in valid_in)
            
            # Actually, the set of valid-in candidates after removing s_out:
            # new_comp_count[x] = comp_count[x] - conflicts[x, s_out]
            # valid if new_comp_count[x] == 0, i.e., comp_count[x] == conflicts[x, s_out]
            
            # For x with comp_count[x] == 0: already valid (in valid_in), no change needed
            # For x with comp_count[x] == 1: valid iff conflicts[x, s_out] == 1
            
            # Candidates that become valid: those with comp_count == 1 and conflict with s_out
            newly_valid = unselected_mask & (comp_count == 1) & conflicts[:, s_out].astype(bool)
            # Also include already-valid ones
            all_valid_after_removal = valid_in | newly_valid
            
            if not np.any(all_valid_after_removal):
                continue
            
            # Among valid candidates, pick one that minimizes future conflicts
            # Score: number of other selected vertices it conflicts with (excluding s_out)
            # For candidate u: future_conflicts = comp_count[u] - conflicts[u, s_out]
            #   If u was in valid_in: future_conflicts = 0
            #   If u was in newly_valid: future_conflicts = 1 - 1 = 0
            # So all have 0 future conflicts with other selected. 
            # To break ties, prefer candidates with lower total degree (fewer conflicts overall)
            
            cand_indices = np.where(all_valid_after_removal)[0]
            
            if len(cand_indices) == 0:
                continue
            
            # Pick the candidate with minimum degree among candidates
            cand_degrees = conflicts[cand_indices].sum(axis=1)
            min_cand_deg = np.min(cand_degrees)
            best_cands = cand_indices[cand_degrees == min_cand_deg]
            u_in = best_cands[rng.integers(len(best_cands))]
            
            # Perform swap
            selected_mask[s_out] = False
            selected_mask[u_in] = True
            current_selected[idx] = u_in
            
            # Update comp_count incrementally
            # Remove s_out contribution: comp_count[x] -= conflicts[x, s_out] for unselected x
            # Add u_in contribution: comp_count[x] += conflicts[x, u_in] for unselected x
            # Note: u_in is now selected, s_out is now unselected
            # s_out becomes unselected, so we need to track its comp_count too
            
            # Update for all vertices that are now unselected (including s_out, excluding u_in)
            # comp_count for unselected x:
            #   new = old - conflicts[x, s_out] + conflicts[x, u_in]
            
            # Vectorized update for all unselected
            # First, mark s_out as unselected and u_in as selected
            # comp_count is only meaningful for unselected vertices
            
            # Update: for all x not selected:
            # comp_count[x] += conflicts[x, u_in] - conflicts[x, s_out]
            not_selected = ~selected_mask
            if np.any(not_selected):
                comp_count[not_selected] += conflicts[not_selected, u_in] - conflicts[not_selected, s_out]
            
            # Recompute valid_in for next iteration
            valid_in = unselected_mask & (comp_count == 0)
            # Update unselected_mask
            unselected_mask = ~selected_mask
            
            improved = True
            break  # Restart with new configuration
    
    return current_selected

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return generate_weight_1_vectors()
    
    # Compute dot products for weight-4 vectors
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (norm^2=4, need dist^2 >= 4 => 8-2*dot >= 4 => dot <= 2)
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS restarts
    mis_deadline = deadline - 30  # Leave 30s for local improvement
    n_restarts = 0
    while time.time() < mis_deadline:
        current_set = greedy_mis(conflicts, mis_deadline, rng)
        n_restarts += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Do local improvement on the best set found
    if len(best_set) > 0 and time.time() < deadline - 5:
        improved_set = local_improve_fast(list(best_set), vectors, conflicts, deadline - 3, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    if not best_set:
        return generate_weight_1_vectors()
    
    result_vectors_w4 = vectors[best_set].astype(np.int64)
    
    # Combine weight-4 and weight-1 vectors
    weight1_vectors = generate_weight_1_vectors()
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors
