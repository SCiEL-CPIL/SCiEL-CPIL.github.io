import time
from math import lcm

def construct():
    deadline = time.time() + 55
    
    denoms = list(range(2, 49))
    
    # Calculate LCM of all denominators
    L = 1
    for d in denoms:
        L = lcm(L, d)
        
    # Split into two halves
    # First half: smaller denominators (larger values), more likely to exceed 1
    # Second half: larger denominators (smaller values), less likely to exceed 1
    mid = 24 # 2..25 (24 items)
    denoms1 = denoms[:mid] # 24 items
    denoms2 = denoms[mid:] # 23 items
    
    n1 = len(denoms1)
    n2 = len(denoms2)
    
    # Precompute integer values
    vals1 = [L // d for d in denoms1]
    vals2 = [L // d for d in denoms2]
    
    # Generate all valid subset sums for half 2
    # Store as (sum, mask)
    # Start with (0, 0)
    sums2 = [(0, 0)]
    
    # Dynamic generation
    for i in range(n2):
        v = vals2[i]
        bit = 1 << i
        # We iterate over a copy of the current list because we are appending to it
        # To be efficient, we can use a list comprehension or extend
        # new_entries = []
        # for s, m in sums2:
        #     new_s = s + v
        #     if new_s < L:
        #         new_entries.append((new_s, m | bit))
        # sums2.extend(new_entries)
        
        # Optimization: local variables
        current_sums = sums2
        new_entries = []
        for s, m in current_sums:
            new_s = s + v
            if new_s < L:
                new_entries.append((new_s, m | bit))
        sums2.extend(new_entries)
        
    # Sort sums2 by sum
    # We only care about the sum for binary search, but need mask for reconstruction
    sums2.sort(key=lambda x: x[0])
    
    # Extract sorted values for binary search
    # To save memory/time in binary search, we can just use the list of tuples
    # bisect works on keys
    sorted_sums2 = [x[0] for x in sums2]
    # We need to map back to mask. 
    # If we have duplicate sums, bisect might return any of them. That's fine.
    # But we need the mask corresponding to the found sum.
    # It's easier to keep the list of tuples and bisect on it with a key?
    # Python's bisect doesn't support key directly.
    # So we keep sorted_sums2 and a separate list of masks?
    # Or we can just bisect on sorted_sums2 and then find the mask?
    # If there are duplicates, we might pick the wrong mask if we don't align them.
    # But the sum is the same, so the "value" is the same. The mask just needs to be *a* mask that produces that sum.
    # So we can create a list of masks aligned with sorted_sums2.
    
    masks2 = [x[1] for x in sums2]
    
    # Generate valid subset sums for half 1
    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0
    
    # We can iterate through all masks of half 1
    # But we can prune if sum >= L
    # Generating sums for half 1 can also be done via DP with pruning
    sums1 = [(0, 0)]
    for i in range(n1):
        v = vals1[i]
        bit = 1 << i
        current_sums = sums1
        new_entries = []
        for s, m in current_sums:
            new_s = s + v
            if new_s < L:
                new_entries.append((new_s, m | bit))
        sums1.extend(new_entries)
        
    # Now for each sum in sums1, find best match in sums2
    import bisect
    
    for s1, m1 in sums1:
        # We want s1 + s2 < L  => s2 < L - s1
        target = L - s1
        # Find largest s2 < target
        # bisect_left(sorted_sums2, target) gives index of first element >= target
        idx = bisect.bisect_left(sorted_sums2, target) - 1
        
        if idx < 0:
            continue
            
        s2 = sorted_sums2[idx]
        total = s1 + s2
        
        if total > best_sum:
            best_sum = total
            best_mask1 = m1
            best_mask2 = masks2[idx]
            
        if time.time() > deadline:
            break
            
    # Construct indices
    indices = []
    
    # Half 1 indices
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    # Half 2 indices
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    if not indices:
        return [0]
        
    return indices
