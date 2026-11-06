import time

def construct():
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    # inverses[i] is the inverse of denominators[i] mod MOD
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)
        
    # Total number of subsets is 2^10 = 1024
    # We use bitmask to represent subsets
    # mask bit i corresponds to denominator denominators[i]
    
    max_mask = 1 << MAX_DENOM
    
    # We want to find the shortest subset for each residue.
    # best_mask[r] will store the bitmask of the shortest subset summing to r.
    # best_len[r] will store the length of that subset.
    
    best_mask = [None] * MOD
    best_len = [MAX_DENOM + 1] * MOD  # Initialize with max possible length + 1
    
    # Iterate through all non-empty masks
    # We can iterate in any order if we keep track of min length.
    # Or we can iterate by length to potentially stop early if we find a solution for all,
    # but since we need the GLOBAL min max, we need to know the min length for EACH residue.
    # So we must check all masks or at least all masks up to the point where we are sure
    # no shorter mask exists for any residue.
    # Actually, just checking all 1023 masks is trivial.
    
    for mask in range(1, max_mask):
        # Calculate sum and length for this mask
        # Optimization: Use DP or simple loop. For 10 bits, simple loop is fine.
        # Or use lsb trick.
        
        # Calculate sum
        s = 0
        l = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s = (s + inverses[idx]) % MOD
            l += 1
            m ^= lsb
            
        # Update best for residue s
        if l < best_len[s]:
            best_len[s] = l
            best_mask[s] = mask
        elif l == best_len[s]:
            # Tie-breaker: keep the one with smaller mask for determinism, 
            # or just keep the first one. Doesn't affect max length.
            pass

    # Determine the minimum possible maximum length (L_star)
    # This is the max of best_len[r] for all r
    L_star = 0
    for r in range(MOD):
        if best_len[r] > L_star:
            L_star = best_len[r]
            
    # Construct the final result
    # For each residue r, convert the best_mask[r] back to a list of denominators
    results = [None] * MOD
    for r in range(MOD):
        mask = best_mask[r]
        if mask is None:
            # Fallback, should not happen
            results[r] = [1]
        else:
            combo = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                combo.append(denominators[idx])
                m ^= lsb
            results[r] = combo
            
    return results
