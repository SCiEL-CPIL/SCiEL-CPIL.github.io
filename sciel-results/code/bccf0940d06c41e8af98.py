import time
import bisect
from math import lcm

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    # 23 and 24 is a good split for 47
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Generate left sums
    # Using list comprehension or loop?
    # Loop with pre-allocation is fast
    left_sums = [0] * (1 << left_size)
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(1 << i):
            left_sums[bit | mask] = left_sums[mask] + w
            
    # Generate right sums and masks
    # We store (sum, mask) to avoid a separate list for sums
    right_pairs = [(0, 0)] * (1 << right_size)
    # Actually, initializing with tuples is slow.
    # Let's use two lists and combine? Or just generate sums and masks separately?
    # Generating (sum, mask) directly is tricky with the bit trick.
    
    # Standard bit trick generates sums. We need masks too.
    # mask is just the index in the array if we fill it in order?
    # No, the bit trick fills indices 0..2^n-1.
    # So right_sums[i] corresponds to mask i.
    
    right_sums = [0] * (1 << right_size)
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(1 << i):
            right_sums[bit | mask] = right_sums[mask] + w
    
    # Create list of (sum, mask)
    # This takes time but is necessary for sorting by sum
    right_pairs = [(right_sums[m], m) for m in range(1 << right_size)]
    right_pairs.sort()
    
    best_deficit = L
    best_left_mask = 0
    best_right_mask = 0
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    for mask in range(1 << left_size):
        # Time check
        if mask % 1000000 == 0 and time.time() > deadline:
            break
            
        ls = left_sums[mask]
        if ls >= L:
            continue
            
        limit = L - ls
        
        # Find right sum < limit
        # bisect_left on right_pairs with key (limit, -1)
        # Since masks are >= 0, (limit, -1) will be placed before any (limit, mask)
        idx = bisect.bisect_left(right_pairs, (limit, -1)) - 1
        
        if idx >= 0:
            rs, rm = right_pairs[idx]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = mask
                best_right_mask = rm
                
    # Reconstruct indices
    result_indices = []
    for i in range(left_size):
        if best_left_mask & (1 << i):
            result_indices.append(i)
    for i in range(right_size):
        if best_right_mask & (1 << i):
            result_indices.append(left_size + i)
            
    if not result_indices:
        result_indices = [0]
        
    return result_indices
