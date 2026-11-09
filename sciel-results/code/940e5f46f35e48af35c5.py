<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 12
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    # Greedy construction
    chosen = []
    
    # To speed up, we can precompute the list of masks sorted by size or something?
    # Not strictly necessary for 4096 items.
    
    for _ in range(40):
        best_gain = -1
        best_mask = -1
        
        for m in all_masks:
            if m in chosen:
                continue
            
            gain = 0
            for c in chosen:
                # Check strict subset relations
                if m != c:
                    if (m & c) == m:
                        # m is subset of c
                        gain += 1
                    elif (m & c) == c:
                        # c is subset of m
                        gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        
        chosen.append(best_mask)
        
    # Convert masks to lists of integers
    result = []
    for m in chosen:
        s = []
        for i in range(n):
            if (m >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    n = 12
    total_masks = 1 << n
    all_masks = np.array(range(total_masks), dtype=np.uint32)
    
    def total_score_np(masks_np):
        # masks_np is shape (40,)
        m1 = masks_np[:, np.newaxis]
        m2 = masks_np[np.newaxis, :]
        comp = ((m1 & m2) == m1) | ((m1 & m2) == m2)
        return np.sum(np.triu(comp, 1))

    best_global_score = -1
    best_global_masks = None
    
    start_time = time.time()
    
    while time.time() - start_time < 55:
        # Initialize with random distinct masks
        indices = random.sample(range(total_masks), 40)
        current_masks = all_masks[indices].copy()
        current_score = total_score_np(current_masks)
        
        # Local Search
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap_index = -1
            best_new_mask_val = -1
            
            for i in range(40):
                old_mask = current_masks[i]
                other_masks = np.delete(current_masks, i)
                
                m1 = all_masks[:, np.newaxis]
                m2 = other_masks[np.newaxis, :]
                
                comp = ((m1 & m2) == m1) | ((m1 & m2) == m2)
                gains = np.sum(comp, axis=1)
                
                # Mask out masks that are already in current_masks
                in_set = np.zeros(total_masks, dtype=bool)
                in_set[current_masks] = True
                gains[in_set] = -1
                
                max_gain = np.max(gains)
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_new_mask_val = np.argmax(gains)
                    best_swap_index = i
            
            if best_swap_index != -1:
                current_masks[best_swap_index] = all_masks[best_new_mask_val]
                current_score = total_score_np(current_masks)
                improved = True

        if current_score > best_global_score:
            best_global_score = current_score
            best_global_masks = current_masks.copy()

    result = []
    for m in best_global_masks:
        m = int(m)
        s = []
        for i in range(n):
            if (m >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
