import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2 (need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to int8 for memory efficiency in operations
    conflicts_int = conflicts.astype(np.int8)
    
    best_selected = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple greedy MIS restarts with different strategies
    mis_deadline = deadline - 20  # Leave 20s for local improvement
    
    # Strategy 1: Greedy with min-degree (multiple restarts)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > mis_deadline:
            break
        rng2 = np.random.default_rng(seed)
        
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts_int.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            if time.time() > mis_deadline:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng2.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflicts_int[chosen] == 1)[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts_int[rem][:, to_remove].sum(axis=1)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
    
    # Strategy 2: Greedy with random ordering (multiple restarts)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > mis_deadline:
            break
        rng2 = np.random.default_rng(seed)
        
        alive = np.ones(N4, dtype=bool)
        selected = []
        order = rng2.permutation(N4)
        for idx in order:
            if time.time() > mis_deadline:
                break
            if alive[idx]:
                selected.append(idx)
                neighbors = np.where(conflicts_int[idx] == 1)[0]
                to_remove = np.unique(np.concatenate(([idx], neighbors)))
                alive[to_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
    
    # Strategy 3: Greedy with max-degree (pick most constrained first)
    for seed in [42, 123, 456]:
        if time.time() > mis_deadline:
            break
        rng2 = np.random.default_rng(seed)
        
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts_int.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            if time.time() > mis_deadline:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng2.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflicts_int[chosen] == 1)[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts_int[rem][:, to_remove].sum(axis=1)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
    
    if best_selected is None:
        return vectors_v1
    
    # Local search: try to swap out one vector for two (1-for-2 improvement)
    selected_arr = np.array(best_selected)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Shuffle to avoid bias
        rng.shuffle(selected_arr)
        
        for i in range(len(selected_arr)):
            if time.time() > deadline - 5:
                break
            
            remove_idx = selected_arr[i]
            
            # Find candidates that don't conflict with remaining selected
            remaining_mask = np.zeros(N4, dtype=bool)
            remaining_mask[selected_arr] = True
            remaining_mask[remove_idx] = False
            
            # Check conflicts with remaining selected
            if np.any(remaining_mask):
                has_conflict = conflicts_int[:, remaining_mask].any(axis=1)
            else:
                has_conflict = np.zeros(N4, dtype=bool)
            
            # Candidates: not in selected, not conflicting with remaining
            selected_mask = np.zeros(N4, dtype=bool)
            selected_mask[selected_arr] = True
            candidates = ~has_conflict & ~selected_mask
            
            if not np.any(candidates):
                continue
            
            cand_indices = np.where(candidates)[0]
            
            # Try to find 2 non-conflicting candidates
            # Use greedy: pick first, then find second that doesn't conflict
            rng.shuffle(cand_indices)
            
            found_two = False
            temp_selected = []
            
            for cand in cand_indices:
                if len(temp_selected) >= 2:
                    break
                # Check if cand conflicts with any in temp_selected
                ok = True
                for ts in temp_selected:
                    if conflicts_int[cand, ts]:
                        ok = False
                        break
                if ok:
                    temp_selected.append(cand)
            
            if len(temp_selected) >= 2:
                # Perform the swap
                new_selected = list(selected_arr)
                new_selected.pop(i)
                new_selected.extend(temp_selected)
                selected_arr = np.array(new_selected)
                improved = True
                best_size = len(selected_arr)
                break
    
    result_v4 = vectors_v4[selected_arr]
    
    # Combine V4 and V1
    # All V1 vectors are compatible with all V4 vectors (dot product is -2, 0, or 2)
    # Among V1 vectors: can take both +2 and -2 for each position (dot = -4, ok)
    # So all 22 V1 vectors are valid
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
