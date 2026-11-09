"""Optimized construction of large 4-uniform sunflower-free family."""

def construct():
    from itertools import combinations
    import time
    import numpy as np
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)  # 1820
    
    # Convert to bitmasks
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)
    
    # Greedy with vectorized conflict check
    # For a candidate c and chosen sets A_1, ..., A_n:
    #   I_i = A_i & c (intersection of A_i with c)
    #   K_{i,j} = A_i & A_j (pairwise intersections among chosen)
    #   Conflict exists if there are i, j with K_{i,j} == I_i AND I_j == I_i
    #   i.e., A_i & A_j == A_i & c AND A_j & c == A_i & c
    
    def check_conflict(c_mask, chosen_masks):
        """Check if c_mask conflicts with any pair in chosen_masks."""
        n = len(chosen_masks)
        if n < 2:
            return False
        
        # I_i = chosen[i] & c_mask for all i
        I = chosen_masks & c_mask  # shape (n,)
        
        # K_{i,j} = chosen[i] & chosen[j]
        # Vectorized: K = chosen_masks[:, None] & chosen_masks[None, :]
        K = chosen_masks[:, None] & chosen_masks[None, :]  # shape (n, n)
        
        # For each i, we need to check if there exists j != i such that:
        #   K[i,j] == I[i] AND I[j] == I[i]
        # I[i] is a scalar for each i.
        # K[i,j] == I[i] means K[i,j] == I[i]
        # I[j] == I[i] means I[j] == I[i]
        
        # Create mask for I[j] == I[i]:
        # SameI[i,j] = (I[i] == I[j])
        SameI = (I[:, None] == I[None, :])  # shape (n, n)
        
        # K[i,j] == I[i]:
        # K is (n,n), I[i] needs to be broadcast to row i
        K_eq_I = (K == I[:, None])  # shape (n, n), K_eq_I[i,j] = (K[i,j] == I[i])
        
        # Conflict if K_eq_I[i,j] AND SameI[i,j] AND i != j
        diag = np.eye(n, dtype=bool)
        conflict_mask = K_eq_I & SameI & ~diag
        
        return np.any(conflict_mask)
    
    # Greedy with multiple random restarts
    best_size = 0
    best_indices = []
    
    # First try sequential order
    chosen_indices = []
    chosen_masks_list = []
    
    for idx in range(n_total):
        if time.time() > deadline:
            break
        cm = masks[idx]
        
        if len(chosen_masks_list) < 2:
            chosen_indices.append(idx)
            chosen_masks_list.append(cm)
            continue
        
        chosen_arr = np.array(chosen_masks_list, dtype=np.int64)
        if not check_conflict(cm, chosen_arr):
            chosen_indices.append(idx)
            chosen_masks_list.append(cm)
    
    best_size = len(chosen_indices)
    best_indices = list(chosen_indices)
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 5 and restarts < 20:
        order = rng.permutation(n_total)
        chosen_indices_r = []
        chosen_masks_r = []
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            cm = masks[idx]
            
            if len(chosen_masks_r) < 2:
                chosen_indices_r.append(idx)
                chosen_masks_r.append(cm)
                continue
            
            chosen_arr = np.array(chosen_masks_r, dtype=np.int64)
            if not check_conflict(cm, chosen_arr):
                chosen_indices_r.append(idx)
                chosen_masks_r.append(cm)
        
        restarts += 1
        if len(chosen_indices_r) > best_size:
            best_size = len(chosen_indices_r)
            best_indices = list(chosen_indices_r)
    
    return [sorted(all_sets[i]) for i in best_indices]
