import time

def construct():
    n = 5040
    target = 4979
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin

    # Generate divisors of n
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    
    divs.sort(reverse=True)
    
    # Filter: only divisors <= target can be used
    divs = [d for d in divs if d <= target]
    num_divs = len(divs)
    
    # Bitset DP: bitset[k] is a big int where bit s is set if sum s is reachable
    # using exactly k divisors from the processed set
    # We'll process divisors one by one, updating bitsets
    
    # First, let's figure out a reasonable max k
    # The largest divisor is 5040, but we filtered to <= 4979
    # Max divisors needed: worst case all 1s, but we have distinct divisors
    # Sum of all divisors of 5040 is quite large, so k won't be huge
    # Let's estimate: sum of top divisors
    max_k = min(num_divs, 20)  # Conservative upper bound
    
    # bitsets[k] = set of reachable sums using exactly k divisors
    # Start: bitsets[0] has bit 0 set (sum 0 with 0 divisors)
    bitsets = [0] * (max_k + 1)
    bitsets[0] = 1  # bit 0 set
    
    # Track which divisors contributed to each state for backtracking
    # parent[k][s] = index of last divisor added to reach sum s with k divisors
    # This is memory intensive, so we'll use a different approach:
    # Store for each (k, s) the last divisor index used
    
    # Actually, for backtracking, let's store the choice at each step
    # choice[div_idx][k][s] = True if divs[div_idx] was used to reach (k, s)
    # This is too much memory. Alternative: recompute during backtracking
    
    # Better approach: process divisors, and for backtracking, 
    # we can re-run the DP in reverse or store parent pointers efficiently
    
    # Let's use: for each divisor processed, store which (k, s) states were newly reached
    # and by which divisor. We'll store parent[k][s] = div_idx
    
    # Memory: (max_k+1) * (target+1) ints = 21 * 4980 ≈ 100k ints, fine
    parent = [[-1] * (target + 1) for _ in range(max_k + 1)]
    
    # Process each divisor
    for div_idx, d in enumerate(divs):
        if time.time() > deadline:
            break
        
        # Update bitsets in reverse order of k to avoid using d multiple times
        for k in range(min(div_idx + 1, max_k - 1), -1, -1):
            # New sums reachable: old bitsets[k] shifted left by d
            new_bits = (bitsets[k] << d) & ((1 << (target + 1)) - 1)
            
            # For each newly set bit in new_bits that wasn't in bitsets[k+1],
            # record parent
            newly_reached = new_bits & ~bitsets[k + 1]
            
            if newly_reached:
                # Find which sums are newly reached
                # Iterate through set bits
                mask = newly_reached
                while mask:
                    # Get lowest set bit
                    low_bit = mask & (-mask)
                    s = low_bit.bit_length() - 1
                    parent[k + 1][s] = div_idx
                    mask ^= low_bit
                
                bitsets[k + 1] |= newly_reached
    
    # Check if target is reachable with any k
    best_k = -1
    for k in range(1, max_k + 1):
        if (bitsets[k] >> target) & 1:
            best_k = k
            break
    
    if best_k == -1:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divs:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    # Backtrack to find the actual divisors
    result = []
    k = best_k
    s = target
    while k > 0:
        div_idx = parent[k][s]
        if div_idx == -1:
            break
        d = divs[div_idx]
        result.append(d)
        s -= d
        k -= 1
    
    # Verify
    if sum(result) == target and len(result) == best_k:
        return result
    
    # Fallback if backtracking failed
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
