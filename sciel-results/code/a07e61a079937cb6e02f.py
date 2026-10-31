import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
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
    
    # Compute conflicts
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS attempts
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        alive = np.ones(N4, dtype=bool)
        selected_indices = []
        degrees = conflicts.sum(axis=1).astype(np.int32).copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Random tie-breaking for min degree
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected_indices.append(chosen)
            
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        # Post-processing: Try to add back removed nodes?
        # Simple local search: iterate through unselected nodes, if no neighbor in selected, add.
        selected_set = set(selected_indices)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Check unselected nodes
            unselected = np.where(~np.isin(np.arange(N4), selected_set))[0]
            for u in unselected:
                # Check if u has any neighbor in selected_set
                # neighbors of u are conflicts[u]
                # We need to check if any of them are in selected_set
                # Efficient check:
                # Create a boolean mask of selected
                # Then check conflicts[u] & selected_mask
                # But this is O(N) per u.
                # Better: maintain a count of selected neighbors for each node?
                # That's complex.
                # Given time, skip complex local search.
                pass
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_selected = np.zeros(N4, dtype=bool)
            best_selected[selected_indices] = True

    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
