import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate integer vectors
    # We focus on small norms to maximize the count
    # Coordinates in {-2, -1, 1, 2}
    # We will generate all vectors with L2 norm squared <= 10
    candidates = []
    
    # Helper to add vector if not zero
    def add_candidate(v):
        norm_sq = np.dot(v, v)
        if norm_sq > 0 and norm_sq <= 10:
            candidates.append(v)

    # Generate all combinations of indices and values
    # To be efficient, we iterate over possible non-zero counts
    # 1 non-zero
    for i in range(D):
        for val in [-2, -1, 1, 2]:
            v = [0]*D
            v[i] = val
            add_candidate(v)
            
    # 2 non-zeros
    for i in range(D):
        for j in range(i+1, D):
            for vi in [-2, -1, 1, 2]:
                for vj in [-2, -1, 1, 2]:
                    v = [0]*D
                    v[i] = vi
                    v[j] = vj
                    add_candidate(v)
                    
    # 3 non-zeros
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for vi in [-1, 1]: # Restrict to +-1 for 3 non-zeros to keep count manageable and norms low
                    for vj in [-1, 1]:
                        for vk in [-1, 1]:
                            v = [0]*D
                            v[i] = vi
                            v[j] = vj
                            v[k] = vk
                            add_candidate(v)
                            
    # 4 non-zeros (all +-1)
    # Might be too many, but let's check count: C(11,4)*16 = 330 * 16 = 5280. Manageable.
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for vi in [-1, 1]:
                        for vj in [-1, 1]:
                            for vk in [-1, 1]:
                                for vl in [-1, 1]:
                                    v = [0]*D
                                    v[i] = vi
                                    v[j] = vj
                                    v[k] = vk
                                    v[l] = vl
                                    add_candidate(v)

    # Remove duplicates (though generation should be unique)
    # Convert to numpy array for speed
    if not candidates:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Sort by norm
    sorted_indices = np.argsort(norms_sq)
    candidates = candidates[sorted_indices]
    norms_sq = norms_sq[sorted_indices]
    
    # Greedy Construction
    selected = []
    selected_norms_sq = []
    max_norm_sq = 0
    
    for idx in range(len(candidates)):
        if time.time() > deadline:
            break
        v = candidates[idx]
        v_norm_sq = norms_sq[idx]
        
        # Check if adding v violates constraints
        # New max norm is max(max_norm_sq, v_norm_sq)
        new_max_norm_sq = max(max_norm_sq, v_norm_sq)
        
        # Check distance to all existing selected vectors
        compatible = True
        for s_idx in range(len(selected)):
            u = selected[s_idx]
            diff = v - u
            dist_sq = np.dot(diff, diff)
            if dist_sq < new_max_norm_sq:
                compatible = False
                break
                
        if compatible:
            selected.append(v)
            selected_norms_sq.append(v_norm_sq)
            max_norm_sq = new_max_norm_sq

    # Local Search to improve
    # We try to swap out a vector for a better one, or add new ones
    # Objective: Maximize len(selected)
    
    # Precompute distances between candidates? 
    # If candidates are ~5000, 5000x5000 matrix is 25M floats. 200MB. Might be too much.
    # We will compute on the fly.
    
    current_size = len(selected)
    best_size = current_size
    best_selected = [v.copy() for v in selected]
    
    # Set of selected indices for fast lookup? 
    # We work with the list 'selected'
    
    # To speed up local search, we only consider candidates that are "close" in norm to the current max
    # Or just random sampling
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Try to add a random candidate
        if random.random() < 0.5:
            idx = random.randint(0, len(candidates)-1)
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            # Check if v is already in selected
            if any(np.array_equal(v, s) for s in selected):
                continue
                
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            compatible = True
            for s in selected:
                diff = v - s
                dist_sq = np.dot(diff, diff)
                if dist_sq < new_max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                max_norm_sq = new_max_norm_sq
                current_size += 1
                improved = True
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_selected = [v.copy() for v in selected]
                continue
                
        # 2. Try to swap a vector
        # Pick a random index in selected
        s_idx = random.randint(0, len(selected)-1)
        old_v = selected[s_idx]
        
        # Find a replacement
        # Ideally, a vector that allows adding more others, or just has a "better" property
        # For now, just try to find a random vector that fits if we remove old_v
        # But this is complex. Simple swap: remove old, add new.
        
        # Try to remove old_v and add a random candidate
        candidate_idx = random.randint(0, len(candidates)-1)
        new_v = candidates[candidate_idx]
        
        if np.array_equal(old_v, new_v):
            continue
            
        # Temporarily remove old_v
        selected[s_idx] = None # Placeholder
        
        # Check if new_v fits with the rest
        # The rest is selected with s_idx removed
        rest = [x for i, x in enumerate(selected) if x is not None]
        
        new_max_norm_sq = 0
        for r in rest:
            r_norm = np.dot(r, r)
            if r_norm > new_max_norm_sq:
                new_max_norm_sq = r_norm
        
        new_v_norm_sq = norms_sq[candidate_idx]
        total_max_norm_sq = max(new_max_norm_sq, new_v_norm_sq)
        
        compatible = True
        for r in rest:
            diff = new_v - r
            dist_sq = np.dot(diff, diff)
            if dist_sq < total_max_norm_sq:
                compatible = False
                break
        
        if compatible:
            # Accept the swap
            selected[s_idx] = new_v
            # Update max_norm_sq
            max_norm_sq = 0
            for s in selected:
                n = np.dot(s, s)
                if n > max_norm_sq:
                    max_norm_sq = n
            improved = True
            no_improve_count = 0
            # Size remains same, so best_size doesn't update unless we track size separately
            # But maybe this swap enables future additions?
            # We don't explicitly track potential, but we keep the state.
            if len(selected) > best_size:
                 best_size = len(selected)
                 best_selected = [v.copy() for v in selected]
        else:
            # Revert
            selected[s_idx] = old_v
            
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
        # Occasionally, if we are stuck, try a random restart of the local search part?
        # Or just accept the stagnation.
        
    # Return the best found
    if best_selected:
        return np.array(best_selected, dtype=np.int64)
    else:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

# EVOLVE-BLOCK-END
