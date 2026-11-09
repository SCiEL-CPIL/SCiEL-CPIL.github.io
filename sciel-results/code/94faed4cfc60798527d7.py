import time
import bisect
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    # Split into two halves
    half1 = n // 2
    half2 = n - half1
    
    w1 = weights[:half1]
    w2 = weights[half1:]
    
    # Generate sums for half 1
    n1 = len(w1)
    size1 = 1 << n1
    sums1 = [0] * size1
    for i in range(n1):
        bit = 1 << i
        w = w1[i]
        for mask in range(bit):
            sums1[bit | mask] = sums1[mask] + w
            
    # Generate sums for half 2
    n2 = len(w2)
    size2 = 1 << n2
    sums2 = [0] * size2
    for i in range(n2):
        bit = 1 << i
        w = w2[i]
        for mask in range(bit):
            sums2[bit | mask] = sums2[mask] + w
            
    # Prepare half 2 for binary search
    pairs2 = [(sums2[mask], mask) for mask in range(size2)]
    pairs2.sort()
    
    # Extract just the sums for bisect
    sums2_sorted = [p[0] for p in pairs2]
    
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through half 1 sums
    for mask1 in range(size1):
        if time.time() > deadline:
            break
            
        s1 = sums1[mask1]
        
        # We need s1 + s2 < L
        # s2 < L - s1
        limit = L - s1
        
        # Find largest s2 < limit
        idx = bisect.bisect_left(sums2_sorted, limit)
        
        if idx > 0:
            s2 = sums2_sorted[idx - 1]
            total = s1 + s2
            deficit = L - total
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = mask1
                best_mask2 = pairs2[idx - 1][1]
                
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            result_indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            result_indices.append(half1 + i)
            
    # Ensure non-empty
    if not result_indices:
        result_indices = [0]
        
    # --- LOCAL SEARCH POLISH ---
    # Convert to a boolean list for easier manipulation
    in_set = [False] * n
    for idx in result_indices:
        in_set[idx] = True
        
    # Calculate current sum as a Fraction for precision in local search
    # Using Fractions is slower but we only do this for a limited number of steps
    current_sum = Fraction(0, 1)
    for i in range(n):
        if in_set[i]:
            current_sum += Fraction(1, denominators[i])
            
    best_sum_frac = current_sum
    best_indices = list(result_indices)
    
    # Local search: try adding, removing, or swapping elements
    # We want to maximize the sum while keeping it < 1
    # i.e., minimize the deficit 1 - sum
    
    # Precompute reciprocal values for quick access
    reciprocals = [Fraction(1, d) for d in denominators]
    
    max_steps = 500000
    step = 0
    improved = True
    while improved and step < max_steps and time.time() < deadline - 5:
        improved = False
        step += 1
        
        # Try adding each element not in the set
        for i in range(n):
            if not in_set[i]:
                new_sum = current_sum + reciprocals[i]
                if new_sum < 1:
                    # This is a valid move that increases the sum
                    in_set[i] = True
                    current_sum = new_sum
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try removing each element in the set (only if it allows adding a larger one? 
        # No, simple removal decreases sum, which is bad for minimizing deficit.
        # Unless removing allows adding multiple larger ones. 
        # But simple 1-for-1 swap is better.)
        
        # Try 1-for-1 swaps
        found_swap = False
        for i_out in range(n):
            if not in_set[i_out]:
                continue
            for i_in in range(n):
                if in_set[i_in]:
                    continue
                # Swap i_out with i_in
                delta = reciprocals[i_in] - reciprocals[i_out]
                new_sum = current_sum + delta
                if new_sum < 1:
                    # Valid swap that increases sum
                    in_set[i_out] = False
                    in_set[i_in] = True
                    current_sum = new_sum
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if improved:
            continue
            
        # Try 1-for-2 swaps? (Remove 1, add 2)
        # This is expensive O(n^3) per step if naive.
        # Let's try a simpler heuristic: 
        # If we can't improve by 1-add or 1-swap, we might be stuck in a local optimum.
        # Given the time constraints, let's stop here or do a random restart.
        # But random restarts are risky. Let's just break.
        
        break

    # Reconstruct final indices from in_set
    final_indices = []
    for i in range(n):
        if in_set[i]:
            final_indices.append(i)
            
    if not final_indices:
        final_indices = best_indices
        
    return final_indices
