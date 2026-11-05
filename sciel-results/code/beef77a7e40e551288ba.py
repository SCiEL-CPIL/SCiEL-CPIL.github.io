import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s budget to be safe

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two sets
    mid = n // 2
    A_dens = denominators[:mid]
    B_dens = denominators[mid:]
    
    # Generate subset sums for A with pruning
    # List of (sum_float, mask)
    sums_A = [(0.0, 0)]
    for i, d in enumerate(A_dens):
        val = 1.0 / d
        current_len = len(sums_A)
        for k in range(current_len):
            s, m = sums_A[k]
            new_s = s + val
            if new_s < 1.0:
                sums_A.append((new_s, m | (1 << i)))
                
    # Generate subset sums for B with pruning
    sums_B = [(0.0, 0)]
    for i, d in enumerate(B_dens):
        val = 1.0 / d
        current_len = len(sums_B)
        for k in range(current_len):
            s, m = sums_B[k]
            new_s = s + val
            if new_s < 1.0:
                sums_B.append((new_s, m | (1 << i)))

    # Sort by sum
    sums_A.sort(key=lambda x: x[0])
    sums_B.sort(key=lambda x: x[0])
    
    # Two-pointer to find best sum < 1.0
    # We want to maximize A_sum + B_sum
    # Iterate A from smallest to largest
    # For each A, find largest B such that A+B < 1
    # As A increases, the max allowed B decreases.
    
    best_candidates = []
    j = len(sums_B) - 1
    
    for i in range(len(sums_A)):
        a_sum, a_mask = sums_A[i]
        
        # Shrink j until valid
        while j >= 0 and a_sum + sums_B[j][0] >= 1.0:
            j -= 1
            
        if j < 0:
            # No valid B for this A or larger A
            # Since A is increasing, all subsequent A will also have no valid B
            break
            
        b_sum, b_mask = sums_B[j]
        total_sum = a_sum + b_sum
        
        # Add to candidates
        # We keep the top 5 to handle float precision issues
        if len(best_candidates) < 5:
            best_candidates.append((total_sum, a_mask, b_mask))
            best_candidates.sort(reverse=True)
        else:
            if total_sum > best_candidates[-1][0]:
                best_candidates.pop()
                best_candidates.append((total_sum, a_mask, b_mask))
                best_candidates.sort(reverse=True)

    # Verify candidates with exact arithmetic
    best_deficit = Fraction(1, 1) # 1 - 0
    best_indices = None
    
    for total_sum, a_mask, b_mask in best_candidates:
        # Reconstruct indices
        indices = []
        for i in range(len(A_dens)):
            if a_mask & (1 << i):
                indices.append(i)
        for i in range(len(B_dens)):
            if b_mask & (1 << i):
                indices.append(mid + i)
                
        if not indices:
            continue
            
        # Calculate exact sum
        exact_sum = sum(Fraction(1, denominators[k]) for k in indices)
        
        if exact_sum < 1:
            deficit = 1 - exact_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_indices = indices
                
        # Time check
        if time.time() > deadline:
            break

    if best_indices is None:
        # Fallback: 1/2 + 1/3 + 1/7 = 41/42 < 1
        # Indices for 2, 3, 7 are 0, 1, 5
        return [0, 1, 5]
        
    return best_indices
