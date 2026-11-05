import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Generate all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    masks = np.array(all_sets, dtype=np.uint32)
    n_sets = len(masks)
    
    best_family_size = 0
    best_family = []
    
    # Multiple restarts
    while time.time() < deadline:
        random.shuffle(all_sets) # Shuffle order? No, shuffle indices for picking
        
        # Re-generate masks if shuffled? 
        # Better to keep masks fixed and shuffle indices
        indices = list(range(n_sets))
        random.shuffle(indices)
        
        conflict_counts = np.zeros(n_sets, dtype=np.int32)
        available_mask = np.ones(n_sets, dtype=bool)
        family_indices = []
        
        # Greedy construction
        while True:
            # Find valid candidates
            valid_mask = (conflict_counts == 0) & available_mask
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                break
            
            # Pick a random valid index
            # To ensure we explore different parts, pick randomly
            idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
            
            # Add to family
            family_indices.append(idx)
            available_mask[idx] = False
            
            # Update conflict counts
            A = masks[idx]
            for b_idx in family_indices[:-1]: # Exclude the one just added? 
                # The new pairs are (A, B) for all B in family before A was added.
                # So iterate over family_indices[:-1]
                B = masks[b_idx]
                H = A & B
                # Forbidden mask for S: S must not intersect (A\H) or (B\H)
                # And must contain H
                F = (A & ~H) | (B & ~H)
                
                avail_indices = np.where(available_mask)[0]
                if len(avail_indices) == 0:
                    continue
                
                avail_masks = masks[avail_indices]
                cond = ((avail_masks & F) == 0) & ((avail_masks & H) == H)
                
                if np.any(cond):
                    conflict_counts[avail_indices[cond]] += 1
            
            if time.time() > deadline:
                break
        
        current_size = len(family_indices)
        if current_size > best_family_size:
            best_family_size = current_size
            best_family = [masks[i] for i in family_indices]
            
    # Convert best_family back to list of sorted lists
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
