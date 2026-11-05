import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute inverses
    inverses = [0] * 11
    for i in range(1, 11):
        inverses[i] = pow(i, -1, 101)
        
    # Precompute sum and size for all masks using DP
    # mask_sum[mask] = sum of inverses mod 101 for the subset represented by mask
    # mask_size[mask] = number of bits set in mask
    # mask_terms[mask] = list of denominators in the subset
    
    min_size = [11] * 101
    best_subset = [None] * 101
    
    # Iterate through all non-empty masks
    for mask in range(1, 1 << 10):
        # Check deadline
        if time.time() > deadline:
            break
            
        # Extract bits efficiently
        m = mask
        t_list = []
        sum_inv = 0
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            x = idx + 1
            t_list.append(x)
            sum_inv += inverses[x]
            m ^= lsb
            
        res = sum_inv % 101
        size = len(t_list)
        
        if size < min_size[res]:
            min_size[res] = size
            best_subset[res] = t_list
    
    # If we didn't finish (shouldn't happen with 1023 iterations), 
    # fill in any missing residues with a fallback
    for r in range(101):
        if best_subset[r] is None:
            # Fallback: use all 10 denominators (should not happen)
            best_subset[r] = list(range(1, 11))
            min_size[r] = 10
    
    result = [best_subset[r] for r in range(101)]
    
    return result
