import time
import random

def construct():
    start = time.time()
    deadline = 85.0
    
    # Core vertices 0..5
    # Generate masks for sizes 0..3
    masks = []
    for mask in range(1 << 6):
        if bin(mask).count('1') <= 3:
            masks.append(mask)
            
    # masks has 42 elements
    # Identify a pair to remove: one size 2, one size 3 superset
    # Find index of 3 (0b000011) and 7 (0b000111)
    # 3 is in masks, 7 is in masks
    
    target_remove_1 = 3
    target_remove_2 = 7
    
    current_masks = [m for m in masks if m != target_remove_1 and m != target_remove_2]
    
    # current_masks has 40 elements
    
    # Convert to lists for return later, but keep masks for search
    # Local Search using masks
    
    def score_masks(ms):
        cnt = 0
        for i in range(len(ms)):
            for j in range(i+1, len(ms)):
                a, b = ms[i], ms[j]
                if (a & b) == a: cnt += 1
                elif (a & b) == b: cnt += 1
        return cnt

    current_score = score_masks(current_masks)
    
    # Local Search
    # Precompute all possible masks (0..63)
    all_masks = list(range(64))
    
    improved = True
    while improved and (time.time() - start) < deadline:
        improved = False
        current_set = set(current_masks)
        for i in range(40):
            if (time.time() - start) > deadline: break
            old = current_masks[i]
            for new in all_masks:
                if new == old: continue
                if new in current_set: continue
                
                # Calculate delta
                # Remove old, add new
                # This is O(N) per check
                # To speed up, we can precompute degrees?
                # But degrees change.
                
                # Quick check: if new is not comparable to many, skip?
                # Just do the full check, it's fast with bit ops.
                
                old_comparable = 0
                new_comparable = 0
                for j in range(40):
                    if j == i: continue
                    other = current_masks[j]
                    # old vs other
                    if (old & other) == old: old_comparable += 1
                    elif (old & other) == other: old_comparable += 1
                    
                    # new vs other
                    if (new & other) == new: new_comparable += 1
                    elif (new & other) == other: new_comparable += 1
                
                delta = new_comparable - old_comparable
                if delta > 0:
                    current_set.remove(old)
                    current_masks[i] = new
                    current_set.add(new)
                    current_score += delta
                    improved = True
                    break # Restart
    
    # Convert back to lists of ints
    result = []
    for m in current_masks:
        s = [i for i in range(6) if (m >> i) & 1]
        result.append(s)
        
    return result
