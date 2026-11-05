import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 45.0 # Safety margin from 90s limit
    
    # Generate all 4-subsets as bitmasks
    # Elements are 1..16. Map to bits 0..15.
    n = 16
    k = 4
    all_masks = []
    for comb in combinations(range(1, n + 1), k):
        mask = 0
        for x in comb:
            mask |= (1 << (x - 1))
        all_masks.append(mask)
    
    num_masks = len(all_masks)
    
    # Data structures for current family
    family_indices = []
    family_masks = []
    # pair_masks[i] = (I_ab, Forbidden_ab) for pair (family_masks[i], family_masks[j])?
    # Storing pairs is tricky with removals.
    # Let's stick to the O(M^2) check but optimize the inner loop.
    
    def can_add(family_masks, s_mask):
        # Check all pairs in family
        m = len(family_masks)
        for i in range(m):
            a = family_masks[i]
            a_s = a & s_mask
            for j in range(i + 1, m):
                b = family_masks[j]
                # Check if a ∩ b == a ∩ s == b ∩ s
                # a ∩ b must equal a_s
                # b ∩ s must equal a_s
                if (a & b) == a_s and (b & s_mask) == a_s:
                    return False
        return True

    best_size = 0
    best_family = []
    
    # Fallback valid family (just in case)
    # The set {1,2,3,4} is valid (size 1)
    fallback = [[1, 2, 3, 4]]
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # 1. Random Greedy Initialization
        indices = list(range(num_masks))
        random.shuffle(indices)
        
        family_indices = []
        family_masks = []
        
        for idx in indices:
            # Time check inside greedy
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if can_add(family_masks, mask):
                family_indices.append(idx)
                family_masks.append(mask)
                
        current_size = len(family_indices)
        
        if current_size > best_size:
            best_size = current_size
            best_family = family_indices[:]
            if best_size >= 250:
                break
                
        # 2. Local Search / Hill Climbing
        fam_set = set(family_indices)
        avail_indices = [i for i in range(num_masks) if i not in fam_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add a random available set
            if avail_indices:
                random.shuffle(avail_indices)
                # Limit number of attempts to prevent long stalls
                attempts = 0
                for idx in avail_indices:
                    if attempts > 100 or time.time() > deadline:
                        break
                    attempts += 1
                    mask = all_masks[idx]
                    if can_add(family_masks, mask):
                        family_indices.append(idx)
                        family_masks.append(mask)
                        fam_set.add(idx)
                        avail_indices.remove(idx)
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_family = family_indices[:]
                            if best_size >= 250:
                                break
                        break # Restart avail shuffle if added
                
            if current_size >= 250 or time.time() > deadline:
                break
                
            # If we can't add, try a swap
            if family_indices and avail_indices:
                # Pick a random set to remove
                rem_idx_pos = random.randint(0, len(family_indices) - 1)
                rem_idx = family_indices[rem_idx_pos]
                rem_mask = family_masks[rem_idx_pos]
                
                # Remove it
                family_indices.pop(rem_idx_pos)
                family_masks.pop(rem_idx_pos)
                fam_set.discard(rem_idx)
                avail_indices.append(rem_idx)
                current_size -= 1
                
                # Try to add a different set
                random.shuffle(avail_indices)
                added = False
                attempts = 0
                for idx in avail_indices:
                    if attempts > 100 or time.time() > deadline:
                        break
                    attempts += 1
                    mask = all_masks[idx]
                    if can_add(family_masks, mask):
                        family_indices.append(idx)
                        family_masks.append(mask)
                        fam_set.add(idx)
                        avail_indices.remove(idx)
                        current_size += 1
                        added = True
                        if current_size > best_size:
                            best_size = current_size
                            best_family = family_indices[:]
                        break
                
                if not added:
                    # Revert
                    family_indices.append(rem_idx)
                    family_masks.append(rem_mask)
                    fam_set.add(rem_idx)
                    avail_indices.remove(rem_idx)
                    current_size += 1
                    # Don't set improved=True here, as we reverted.
                    # Break to try a different restart or different swap
                    break

    # Return best family found
    if not best_family:
        return fallback
        
    # Convert indices back to sorted lists of elements
    # Bits 0..15 correspond to elements 1..16
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(subset))
        
    return result
