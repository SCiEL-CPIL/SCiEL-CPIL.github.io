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
    
    def check_conflict_fast(c_mask, chosen_masks):
        """Check if c_mask conflicts with any pair in chosen_masks.
        
        Conflict: exist i != j such that:
          A_i & A_j == A_i & c  AND  A_j & c == A_i & c
        Let t = A_i & c. Then need A_j & c == t AND A_i & A_j == t.
        Since t = A_i & c, and A_i & A_j == t means (A_i & A_j) == (A_i & c).
        This implies A_i & (A_j ^ c) == 0, i.e., A_i & A_j & ~c == 0.
        Also A_j & c == A_i & c.
        
        Alternative: group chosen sets by their intersection with c.
        For group G_t = {i : A_i & c == t}, check if any pair i,j in G_t has A_i & A_j == t.
        """
        n = len(chosen_masks)
        if n < 2:
            return False
        
        I = chosen_masks & c_mask  # shape (n,)
        
        # Group indices by their intersection value with c
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[I[i]].append(i)
        
        # For each group with >= 2 members, check pairwise
        for t, idxs in groups.items():
            if len(idxs) < 2:
                continue
            # Check all pairs in this group
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if (chosen_masks[i] & chosen_masks[j]) == t:
                        return True
        return False
    
    def greedy_build(order, start_time, deadline):
        """Greedy build following given order."""
        chosen_indices = []
        chosen_masks_list = []
        
        for idx in order:
            if time.time() > deadline:
                break
            cm = masks[idx]
            
            if len(chosen_masks_list) < 2:
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
                continue
            
            chosen_arr = np.array(chosen_masks_list, dtype=np.int64)
            if not check_conflict_fast(cm, chosen_arr):
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
        
        return chosen_indices, chosen_masks_list
    
    def local_improve(chosen_indices, chosen_masks_list, deadline):
        """Try to improve by swapping: remove one, try to add multiple new ones."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n_chosen = len(chosen_indices)
            chosen_set = set(chosen_indices)
            
            # Try removing each chosen element and see if we can add more
            for remove_pos in range(n_chosen):
                if time.time() > deadline - 2:
                    break
                
                # Remove element at remove_pos
                remaining_indices = chosen_indices[:remove_pos] + chosen_indices[remove_pos+1:]
                remaining_masks = [chosen_masks_list[i] for i in range(n_chosen) if i != remove_pos]
                
                if len(remaining_masks) < 2:
                    continue
                
                remaining_arr = np.array(remaining_masks, dtype=np.int64)
                
                # Try to add elements not in chosen
                added = []
                for idx in range(n_total):
                    if idx in chosen_set:
                        continue
                    if time.time() > deadline - 3:
                        break
                    cm = masks[idx]
                    if not check_conflict_fast(cm, remaining_arr):
                        added.append(idx)
                
                # If we added more than we removed (1), it's an improvement
                if len(added) > 1:
                    new_indices = remaining_indices + added
                    new_masks = remaining_masks + [masks[i] for i in added]
                    chosen_indices = new_indices
                    chosen_masks_list = new_masks
                    improved = True
                    break
        
        return chosen_indices, chosen_masks_list
    
    # First try sequential order
    order_seq = list(range(n_total))
    chosen_indices, chosen_masks_list = greedy_build(order_seq, time.time(), deadline)
    
    best_size = len(chosen_indices)
    best_indices = list(chosen_indices)
    
    # Local improvement on sequential result
    if time.time() < deadline - 10:
        ci, cm = local_improve(chosen_indices, chosen_masks_list, deadline)
        if len(ci) > best_size:
            best_size = len(ci)
            best_indices = list(ci)
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 15 and restarts < 15:
        order = rng.permutation(n_total)
        chosen_r, masks_r = greedy_build(order, time.time(), deadline)
        
        # Local improvement
        if time.time() < deadline - 10 and len(chosen_r) >= best_size - 5:
            ci, cm = local_improve(chosen_r, masks_r, deadline)
            if len(ci) > len(chosen_r):
                chosen_r = ci
        
        restarts += 1
        if len(chosen_r) > best_size:
            best_size = len(chosen_r)
            best_indices = list(chosen_r)
    
    return [sorted(all_sets[i]) for i in best_indices]
