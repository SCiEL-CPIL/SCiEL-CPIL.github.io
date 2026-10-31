import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Generate Pool
    # Weight-1 vectors: ±2 in one coordinate. Norm^2 = 4.
    # These are compatible with all weight-4 vectors (dot product is 0 or ±2, never > 2).
    weight1_vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            weight1_vectors.append(v)
    
    # Weight-4 vectors: 4 non-zero entries, each ±1. Norm^2 = 4.
    weight4_vectors = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            weight4_vectors.append(v)
    
    all_vectors = np.array(weight1_vectors + weight4_vectors, dtype=np.int64)
    n_total = len(all_vectors)
    n_w1 = len(weight1_vectors)
    n_w4 = len(weight4_vectors)
    
    if time.time() > deadline - 10:
        return all_vectors[:10]
    
    # 2. Compute Dot Products
    dots = all_vectors @ all_vectors.T
    
    # 3. Build Conflict Graph
    # Conflict if dot > 2 (since norm^2 = 4, dist^2 = 8 - 2*dot >= 4 => dot <= 2)
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # 4. Initial Independent Set: Start with all weight-1 vectors (they are mutually compatible)
    # Then greedily add weight-4 vectors that are compatible with the current set.
    
    # The weight-1 vectors are at indices 0 to n_w1-1
    initial_set = list(range(n_w1))
    
    # Precompute which weight-4 vectors are compatible with all weight-1 vectors
    # A weight-4 vector is compatible with a weight-1 vector if their dot product <= 2
    # Since weight-1 has ±2 in one position and weight-4 has ±1 in that position (if non-zero),
    # the dot product is ±2 or 0. Both are <= 2, so ALL weight-4 are compatible with ALL weight-1.
    # So we can start with all weight-1 and then greedily add weight-4.
    
    # Greedy add weight-4 vectors
    available = np.ones(n_total, dtype=bool)
    for idx in initial_set:
        available[idx] = False
        available[conflict[idx, :] == 1] = False
    
    # Now greedily add weight-4 vectors
    # Order by degree (ascending) for better results
    degrees = np.sum(conflict, axis=1)
    w4_indices = np.arange(n_w1, n_total)
    w4_order = w4_indices[np.argsort(degrees[w4_indices])]
    
    current_set = set(initial_set)
    for idx in w4_order:
        if not available[idx]:
            continue
        current_set.add(idx)
        available[conflict[idx, :] == 1] = False
        available[idx] = False
    
    best_set = list(current_set)
    best_size = len(best_set)
    
    # 5. Local Improvement: Try to swap out one vector for two or more
    # Simpler local search: Try to add any remaining vector by removing its conflicts
    # Or: Try to replace a low-degree vector with a higher-degree one if it allows more additions
    
    # Let's do a simple hill-climbing: try to add any available vector, if not, try removing
    # one vector from the set to free up space for multiple others.
    
    improved = True
    max_improve_iters = 50
    improve_iter = 0
    
    while improved and improve_iter < max_improve_iters:
        improved = False
        improve_iter += 1
        
        if time.time() > deadline - 15:
            break
        
        current_set = set(best_set)
        current_set_indices = list(current_set)
        
        # Rebuild available mask
        available = np.ones(n_total, dtype=bool)
        for idx in current_set_indices:
            available[conflict[idx, :] == 1] = False
            available[idx] = False
        
        # Try to find a vector not in the set that can be added by removing one conflict
        # For each candidate not in set, check how many conflicts it has with the set
        # If it has exactly 1 conflict, we can swap: remove the conflict, add the candidate
        
        candidates_not_in_set = np.where(~np.isin(np.arange(n_total), current_set_indices))[0]
        
        if len(candidates_not_in_set) == 0:
            break
        
        # For efficiency, check a subset of candidates
        # Sort candidates by how few conflicts they have with the current set
        conflict_counts = np.array([np.sum(conflict[c, current_set_indices]) for c in candidates_not_in_set])
        
        # Try candidates with exactly 1 conflict first
        single_conflict_idx = candidates_not_in_set[conflict_counts == 1]
        
        if len(single_conflict_idx) > 0:
            # Try to swap: for each such candidate, find the one conflict and swap
            for cand in single_conflict_idx[:50]:  # Limit to 50 to save time
                if time.time() > deadline - 15:
                    break
                
                # Find the conflict
                conflicts_with_set = [i for i in current_set_indices if conflict[cand, i] == 1]
                if len(conflicts_with_set) != 1:
                    continue
                
                conflict_idx = conflicts_with_set[0]
                
                # Try the swap: remove conflict_idx, add cand
                new_set = list(current_set)
                new_set.remove(conflict_idx)
                new_set.append(cand)
                
                # Check if this allows adding more vectors
                # Rebuild available for new set
                new_available = np.ones(n_total, dtype=bool)
                for idx in new_set:
                    new_available[conflict[idx, :] == 1] = False
                    new_available[idx] = False
                
                # Count how many more we could add
                # This is expensive, so just check if size increased
                # Actually, the swap keeps size the same. We need to check if it enables further additions.
                # For now, just accept the swap if it doesn't decrease size (it doesn't) and hope
                # that it enables better future additions. This is weak.
                
                # Better: after swap, try to greedily add any available vectors
                new_set_size = len(new_set)
                new_available = np.ones(n_total, dtype=bool)
                for idx in new_set:
                    new_available[conflict[idx, :] == 1] = False
                    new_available[idx] = False
                
                # Try to add more
                remaining = np.where(new_available)[0]
                for r in remaining:
                    if new_available[r]:
                        new_set.append(r)
                        new_available[conflict[r, :] == 1] = False
                        new_available[r] = False
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                    improved = True
                    break
        
        if not improved:
            # Try removing a vector to see if we can add more
            # Remove the vector with the highest degree in the current set
            if len(current_set_indices) > 0:
                degrees_in_set = [degrees[i] for i in current_set_indices]
                max_deg_idx = current_set_indices[np.argmax(degrees_in_set)]
                
                new_set = list(current_set)
                new_set.remove(max_deg_idx)
                
                new_available = np.ones(n_total, dtype=bool)
                for idx in new_set:
                    new_available[conflict[idx, :] == 1] = False
                    new_available[idx] = False
                
                # Greedily add
                remaining = np.where(new_available)[0]
                for r in remaining:
                    if new_available[r]:
                        new_set.append(r)
                        new_available[conflict[r, :] == 1] = False
                        new_available[r] = False
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                    improved = True
    
    if not best_set:
        return all_vectors[:1]
    
    result_vectors = all_vectors[best_set]
    return result_vectors
