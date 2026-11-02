import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # 1. Generate candidates
    candidates = []
    for i in range(D):
        for s in [2, -2]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # 2. Dot Product
    D_mat = candidates @ candidates.T
    conflict_mask = (D_mat >= 3)
    np.fill_diagonal(conflict_mask, False)
    
    # 3. Search
    best_set = []
    best_size = 0
    
    # Precompute degrees for potential ordering
    # degrees = np.sum(conflict_mask, axis=1)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Random order
        order = np.random.permutation(N)
        
        current_set = []
        in_set = np.zeros(N, dtype=bool)
        
        # Greedy Phase
        for v_idx in order:
            if in_set[v_idx]:
                continue
            # Check conflict
            if len(current_set) == 0:
                current_set.append(v_idx)
                in_set[v_idx] = True
            else:
                # conflict_mask[v_idx, current_set]
                # current_set is a list of ints
                if not np.any(conflict_mask[v_idx, current_set]):
                    current_set.append(v_idx)
                    in_set[v_idx] = True
                    
        # Second Pass: Add any remaining valid vertices
        # This helps if a vertex was rejected early due to a vertex that was 
        # later rejected? No, vertices in current_set are always valid.
        # A vertex v is rejected if it conflicts with SOME u in current_set.
        # Since current_set only grows, if v conflicts with u, and u is in set, v is invalid.
        # The second pass is redundant if the first pass processed all vertices?
        # No. In the first pass, v is processed at position P.
        # It is rejected if it conflicts with any u added BEFORE P.
        # Vertices added AFTER P are not checked against v.
        # But if v was rejected, it conflicts with some u in set.
        # So v is still invalid.
        # So second pass is NOT needed for validity, but maybe for finding
        # vertices that were "skipped" due to some other reason?
        # No, the condition is just "no conflict with current_set".
        # If it conflicted with current_set at time P, it still conflicts now.
        # So the set is maximal for that order.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    if len(best_set) == 0:
        return np.array([], dtype=np.int64)
        
    return candidates[best_set]
