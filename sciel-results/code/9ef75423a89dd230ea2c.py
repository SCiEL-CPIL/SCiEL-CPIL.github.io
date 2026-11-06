import time
from math import lcm
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Calculate weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    n1 = mid
    n2 = n - mid
    
    # Generate subset sums for first half
    # sums1: list of int
    # masks1: list of int
    sums1 = [0]
    masks1 = [0]
    
    for i in range(n1):
        w = weights[i]
        bit = 1 << i
        current_len = len(sums1)
        # We need to extend the list. 
        # To avoid modifying list while iterating, we can create new lists or append.
        # Appending is O(1) amortized.
        for idx in range(current_len):
            ns = sums1[idx] + w
            # We only care about sums < L because sum >= L is invalid (not strictly below 1)
            # Actually, if sum >= L, it's invalid. So we can prune these?
            # No, we can just not add them to the list if we want to save memory, 
            # but we must be careful: a large sum in H1 might combine with a small sum in H2?
            # No, sums are positive. If sum1 >= L, then sum1 + sum2 >= L. So it's invalid.
            # So we can safely prune sums >= L during generation.
            if ns < L:
                sums1.append(ns)
                masks1.append(masks1[idx] | bit)
                
        if time.time() > deadline - 20:
            break
            
    # Generate subset sums for second half
    sums2 = [0]
    masks2 = [0]
    
    for i in range(n2):
        w = weights[mid + i]
        bit = 1 << i
        current_len = len(sums2)
        for idx in range(current_len):
            ns = sums2[idx] + w
            if ns < L:
                sums2.append(ns)
                masks2.append(masks2[idx] | bit)
                
        if time.time() > deadline - 20:
            break
            
    # Sort second half by sum
    # Create list of indices sorted by sum to keep memory low? 
    # Or just sort tuples. 
    # Sorting 8M tuples is memory heavy. 
    # Let's sort indices.
    
    # Optimization: If one half is too large, we might run out of memory.
    # 8M ints is ~64MB. 8M masks is ~64MB. Total ~128MB. Should be fine.
    
    # Sort sums2 and masks2 together.
    # Using zip is slow. 
    # Let's create a list of indices and sort by key.
    idx2 = list(range(len(sums2)))
    idx2.sort(key=lambda i: sums2[i])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [sums2[i] for i in idx2]
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    # Binary search for each sum in half 1
    for idx1 in range(len(sums1)):
        if time.time() > deadline - 5:
            break
            
        s1 = sums1[idx1]
        m1 = masks1[idx1]
        target = L - 1 - s1  # We want s1 + s2 <= L - 1
        
        # Find largest s2 <= target
        pos = bisect_right(sorted_sums2, target)
        if pos > 0:
            s2 = sorted_sums2[pos - 1]
            # Get corresponding mask
            orig_idx = idx2[pos - 1]
            m2 = masks2[orig_idx]
            
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2
                
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_m1 & (1 << i):
            result_indices.append(i)
            
    for i in range(n2):
        if best_m2 & (1 << i):
            result_indices.append(mid + i)
            
    if not result_indices:
        return [0]
        
    return result_indices
