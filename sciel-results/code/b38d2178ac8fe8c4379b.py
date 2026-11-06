import time
from math import lcm
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Target is 1, which is L in integer units
    # We want max sum < L
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    indices_1 = list(range(mid))
    indices_2 = list(range(mid, n))
    
    weights_1 = [weights[i] for i in indices_1]
    weights_2 = [weights[i] for i in indices_2]
    
    def gen_sums(weights_list):
        # Returns list of (sum, mask) where sum < L
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            bit = 1 << i
            # Filter and add
            # Using list comprehension for speed
            new_sums = [(s + w, m | bit) for s, m in sums if s + w < L]
            sums.extend(new_sums)
            # Optional: check time
            if time.time() > deadline:
                break
        return sums

    sums_1 = gen_sums(weights_1)
    sums_2 = gen_sums(weights_2)
    
    # Sort sums_2 by sum value
    sums_2.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums_2_values = [s[0] for s in sums_2]
    
    best_sum = -1
    best_mask = 0
    
    for s1, m1 in sums_1:
        if time.time() > deadline:
            break
        # We need s2 such that s1 + s2 < L => s2 < L - s1
        # Max s2 is the largest value in sums_2_values < L - s1
        target = L - s1 - 1
        # Find rightmost index in sums_2_values <= target
        # bisect_right gives insertion point to the right of existing entries
        # We want the largest element <= target
        idx = bisect.bisect_right(sums_2_values, target)
        if idx > 0:
            # The element at idx-1 is the largest <= target
            s2, m2 = sums_2[idx-1]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                # Combine masks
                # m2 bits correspond to indices in weights_2 (0..n-mid-1)
                # We need to shift m2 by mid to align with global indices
                best_mask = m1 | (m2 << mid)
                
    # Convert best_mask to indices
    if best_mask == 0:
        # Fallback, should not happen as {0} (1/2) is valid
        return [0]
        
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)
            
    return result_indices
