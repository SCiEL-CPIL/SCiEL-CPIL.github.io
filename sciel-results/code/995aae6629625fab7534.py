import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    half = n // 2
    left_size = half
    right_size = n - half
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Generate all subset sums for left half
    left_sums = [0] * (1 << left_size)
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(1 << i):
            left_sums[bit | mask] = left_sums[mask] + w
            
    # Generate all subset sums for right half
    right_sums = [0] * (1 << right_size)
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(1 << i):
            right_sums[bit | mask] = right_sums[mask] + w
            
    # Sort sums
    left_sums.sort()
    right_sums.sort()
    
    # We also need to keep track of the masks to reconstruct the solution.
    # This is the tricky part.
    # If we just sort the sums, we lose the mask information.
    # We need to sort pairs (sum, mask).
    
    # Let's redo with pairs.
    left_pairs = [(left_sums[mask], mask) for mask in range(1 << left_size)]
    left_pairs.sort()
    
    right_pairs = [(right_sums[mask], mask) for mask in range(1 << right_size)]
    right_pairs.sort()
    
    best_deficit = L
    best_l_mask = 0
    best_r_mask = 0
    
    # Two pointer
    # l_idx goes 0 to end
    # r_idx goes end to 0
    # We want l_sum + r_sum < L
    # Since l_sum increases, r_sum must decrease.
    
    r_idx = len(right_pairs) - 1
    
    for l_idx in range(len(left_pairs)):
        l_sum, l_mask = left_pairs[l_idx]
        
        # If l_sum >= L, no valid r_sum can make it < L (since r_sum >= 0)
        # But we want strictly less than L.
        # If l_sum >= L, break? No, if l_sum is large, r_sum must be small.
        # But r_sum is non-negative.
        # If l_sum >= L, then l_sum + r_sum >= L for all r_sum >= 0.
        # So we can break.
        if l_sum >= L:
            break
            
        # Find largest r_sum such that l_sum + r_sum < L
        # i.e., r_sum < L - l_sum
        # r_idx is the current candidate.
        # We need to decrease r_idx until the condition is met.
        
        while r_idx >= 0 and l_sum + right_pairs[r_idx][0] >= L:
            r_idx -= 1
            
        if r_idx >= 0:
            r_sum, r_mask = right_pairs[r_idx]
            total = l_sum + r_sum
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_l_mask = l_mask
                best_r_mask = r_mask
                
    # Reconstruct
    in_subset = [False] * n
    for i in range(left_size):
        if best_l_mask & (1 << i):
            in_subset[i] = True
    for i in range(right_size):
        if best_r_mask & (1 << i):
            in_subset[left_size + i] = True
            
    result_indices = [i for i in range(n) if in_subset[i]]
    
    if not result_indices:
        result_indices = [0]
        
    return result_indices
