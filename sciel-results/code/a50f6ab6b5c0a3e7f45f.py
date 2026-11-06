import time

def construct():
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)
        
    # Total number of subsets is 2^10 = 1024
    # We use bitmask to represent subsets
    # mask bit i corresponds to denominator denominators[i]
    
    # sums[mask] = sum of inverses of denominators in mask
    # lens[mask] = number of denominators in mask
    # We store the best (shortest) mask for each residue, 
    # but we need to find the global min max length first.
    
    # Step 1: Compute sum and length for all masks
    # mask 0 is empty, sum 0, len 0. But we need non-empty.
    
    max_mask = 1 << MAX_DENOM
    subset_sums = [0] * max_mask
    subset_lens = [0] * max_mask
    
    # Initialize
    # For efficiency, we can compute sum[mask] using sum[mask without lowest bit] + inverse[lowest bit]
    # But 1024 is small, simple loop is fine.
    
    for mask in range(1, max_mask):
        # Find lowest set bit
        lsb = mask & -mask
        # Index of the bit
        # Since MAX_DENOM is 10, we can use bit_length - 1
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        subset_sums[mask] = (subset_sums[prev_mask] + inverses[idx]) % MOD
        subset_lens[mask] = subset_lens[prev_mask] + 1
        
    # Step 2: Determine the minimum possible maximum length (L_star)
    # For each residue r, find the minimum length of any subset that sums to r.
    # min_len_for_residue[r] = min { len(S) | sum(S) == r }
    
    min_len_for_residue = [11] * MOD  # 11 is larger than max possible length 10
    
    for mask in range(1, max_mask):
        r = subset_sums[mask]
        l = subset_lens[mask]
        if l < min_len_for_residue[r]:
            min_len_for_residue[r] = l
            
    # The optimal max length is the maximum of these minimum lengths
    L_star = max(min_len_for_residue)
    
    # Check if any residue is not covered (should not happen)
    if L_star > MAX_DENOM:
        # Fallback, though mathematically unlikely for p=101, N=10
        L_star = MAX_DENOM
        
    # Step 3: Construct the result
    # For each residue r, find a subset with length <= L_star.
    # To be optimal (minimize sum of lengths as well, or just any valid one),
    # we should pick the subset with the minimum length for that residue.
    # If there are ties, pick the one with smallest mask (deterministic).
    
    best_subset_for_residue = [None] * MOD
    
    # We iterate masks again to assign the best subset to each residue
    # We want the shortest subset. If lengths are equal, we can keep the first found or smallest mask.
    # Since we want to minimize the "raw value" which is the max length, 
    # and we already fixed the max length to L_star, 
    # picking the absolute shortest for each row is the best strategy to potentially 
    # minimize any secondary metrics like sum of lengths.
    
    # Initialize with a large length
    current_best_len = [11] * MOD
    current_best_mask = [0] * MOD
    
    for mask in range(1, max_mask):
        r = subset_sums[mask]
        l = subset_lens[mask]
        
        # If this subset is shorter than the current best for residue r
        if l < current_best_len[r]:
            current_best_len[r] = l
            current_best_mask[r] = mask
        elif l == current_best_len[r]:
            # Tie-breaker: smaller mask (arbitrary but deterministic)
            if mask < current_best_mask[r]:
                current_best_mask[r] = mask
                
    # Construct the lists
    results = [None] * MOD
    for r in range(MOD):
        mask = current_best_mask[r]
        if mask == 0:
            # Fallback: should not happen if L_star <= 10
            # Find any subset
            for m in range(1, max_mask):
                if subset_sums[m] == r:
                    mask = m
                    break
            if mask == 0:
                # Last resort, invalid but prevents crash
                results[r] = [1]
                continue
        
        # Convert mask to list of denominators
        subset = []
        for i in range(MAX_DENOM):
            if mask & (1 << i):
                subset.append(denominators[i])
        results[r] = subset
        
    return results
