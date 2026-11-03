import numpy as np
import itertools
import random
import time

D = 11
DEADLINE_S = 115.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    # 1. Generate vectors
    vectors = []
    norms_sq = []
    
    # S2: norm^2 = 2 (two +/-1)
    # Count: C(11,2) * 4 = 220
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                norms_sq.append(2)
                
    # S3: norm^2 = 3 (three +/-1)
    # Count: C(11,3) * 8 = 1320
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            norms_sq.append(3)
            
    # S4: norm^2 = 4 (four +/-1)
    # Count: C(11,4) * 16 = 5280
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            norms_sq.append(4)
            
    # Also include norm^2=4 from one +/-2?
    # v = (2, 0, ...)
    # Count: 11 * 2 = 22
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            vectors.append(v)
            norms_sq.append(4)
            
    N = len(vectors)
    V = np.array(vectors, dtype=np.int64)
    norms_sq_arr = np.array(norms_sq, dtype=np.int64)
    
    # 2. Compute Conflict Matrix
    # Conflict if ||vi - vj||^2 < 4
    # ||vi - vj||^2 = ||vi||^2 + ||vj||^2 - 2 vi.vj
    # We want: ||vi||^2 + ||vj||^2 - 2 vi.vj >= 4
    # So conflict if: 2 vi.vj > ||vi||^2 + ||vj||^2 - 4
    
    # Gram matrix
    G = V @ V.T
    
    # Threshold matrix
    # T[i, j] = (norms_sq[i] + norms_sq[j] - 4) / 2
    # If G[i, j] > T[i, j], conflict.
    
    # Vectorized calculation
    # norms_sq is (N,).
    # sum_norms = norms_sq[:, None] + norms_sq[None, :]
    sum_norms = norms_sq_arr[:, None] + norms_sq_arr[None, :]
    
    # We need 2*G > sum_norms - 4
    # 2*G - sum_norms + 4 > 0
    
    diff = 2 * G - sum_norms + 4
    conflict_mask = diff > 0
    np.fill_diagonal(conflict_mask, False)
    
    # 3. Greedy Independent Set with Local Search
    best_set = []
    best_k = 0
    
    # Initial greedy with random order
    # We can also try to prioritize high-degree nodes or low-degree nodes.
    # For independent set, picking low-degree nodes first is often good.
    degrees = np.sum(conflict_mask, axis=1)
    
    # Sort indices by degree (ascending) for a deterministic baseline
    # But we want random restarts.
    
    def greedy_select(order):
        """
        Greedy selection based on the given order.
        Returns a list of selected indices.
        """
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if time.time() - start_time > DEADLINE_S:
                break
            if selected_mask[idx]:
                continue
            # Check if idx conflicts with any selected
            # conflict_mask[idx] & selected_mask
            # If no conflict, add
            if not np.any(conflict_mask[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
                
        return selected, selected_mask

    def local_search(selected, selected_mask):
        """
        Perform local search on the selected set.
        Tries to remove one and add more.
        """
        improved = True
        while improved and (time.time() - start_time < DEADLINE_S):
            improved = False
            # Try removing a random selected vector
            if not selected:
                break
            
            # Limit the number of removal attempts per pass to avoid infinite loops if no improvement
            # But usually, if we remove and can't add back > 1, we revert.
            # We iterate through a random permutation of selected items to try removals.
            # To be efficient, we just pick a few random ones.
            candidates = random.sample(selected, min(len(selected), 20))
            
            for rem_idx in candidates:
                if time.time() - start_time > DEADLINE_S:
                    break
                
                rem_pos = selected.index(rem_idx)
                
                # Remove it
                selected.pop(rem_pos)
                selected_mask[rem_idx] = False
                
                # Try to add any vector that now fits
                # We can just re-run a mini-greedy on the unselected
                # Or check all unselected
                # Optimization: Only check vectors that were conflicting with rem_idx?
                # Actually, removing rem_idx might allow many vectors.
                # Let's just scan all unselected.
                
                added = []
                # To speed up, we can check against current selected_mask
                for idx in range(N):
                    if selected_mask[idx]:
                        continue
                    if not np.any(conflict_mask[idx] & selected_mask):
                        selected.append(idx)
                        selected_mask[idx] = True
                        added.append(idx)
                        
                if len(added) > 1:
                    improved = True
                    # Update best globally if better
                    if len(selected) > best_k:
                        return True, selected, selected_mask
                else:
                    # Revert
                    for idx in added:
                        selected.remove(idx)
                        selected_mask[idx] = False
                    selected.insert(rem_pos, rem_idx)
                    selected_mask[rem_idx] = True
                    
        return False, selected, selected_mask

    # Strategy:
    # 1. Run min-degree greedy once to get a good baseline.
    # 2. Run random restarts with local search.
    
    # Min-degree greedy
    min_degree_order = np.argsort(degrees)
    selected, selected_mask = greedy_select(min_degree_order)
    if len(selected) > best_k:
        best_k = len(selected)
        best_set = selected[:]
        
    # Try to improve the min-degree solution with local search
    # Note: local_search modifies selected in place. We need a copy.
    sel_copy = selected[:]
    mask_copy = selected_mask.copy()
    _, sel_copy, mask_copy = local_search(sel_copy, mask_copy)
    if len(sel_copy) > best_k:
        best_k = len(sel_copy)
        best_set = sel_copy[:]

    # Random restarts
    # The inspiration suggests that min-degree is a good lower bound.
    # We can use random orders but maybe biased towards low degree?
    # Or just pure random to explore different parts of the space.
    
    for restart in range(200):
        if time.time() - start_time > DEADLINE_S:
            break
            
        # Random order
        order = np.random.permutation(N)
        
        selected, selected_mask = greedy_select(order)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_set = selected[:]
            
        # Local search refinement
        # Only do local search if the initial greedy is reasonably good
        # or if we have time.
        if len(selected) >= best_k - 5: # Close to best
            sel_copy = selected[:]
            mask_copy = selected_mask.copy()
            _, sel_copy, mask_copy = local_search(sel_copy, mask_copy)
            if len(sel_copy) > best_k:
                best_k = len(sel_copy)
                best_set = sel_copy[:]

    # Fallback if no set found (should not happen)
    if not best_set:
        # Return the full S2 set
        s2_indices = list(range(220))
        return V[s2_indices]
        
    return V[best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found {res.shape[0]} vectors")
