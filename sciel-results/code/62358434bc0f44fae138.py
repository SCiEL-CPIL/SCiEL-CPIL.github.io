import time
import math
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = math.lcm(L, d)
        
    # Weights for each denominator
    weights = [L // d for d in denominators]
    
    # Split into two halves
    # n = 47
    # Half 1: 0..22 (23 items)
    # Half 2: 23..46 (24 items)
    n1 = 23
    n2 = n - n1
    
    # Generate subset sums for first half
    # List of (sum, mask)
    sums_A = [(0, 0)]
    for i in range(n1):
        w = weights[i]
        # Double the list: add w to each existing sum
        # Using list comprehension for speed
        new_sums = [(s + w, m | (1 << i)) for s, m in sums_A]
        sums_A.extend(new_sums)
        
        # Time check
        if time.time() > deadline - 20.0:
             # Fallback to greedy if generation is too slow (unlikely for 2^23)
             return _heuristic_fallback(denominators)

    # Generate subset sums for second half
    sums_B = [(0, 0)]
    for i in range(n2):
        w = weights[n1 + i]
        new_sums = [(s + w, m | (1 << i)) for s, m in sums_B]
        sums_B.extend(new_sums)
        
        if time.time() > deadline - 15.0:
             return _heuristic_fallback(denominators)

    # Sort sums_B by sum
    sums_B.sort(key=lambda x: x[0])
    
    # Extract sums and masks for binary search
    # We need to search for the largest sum < target
    # bisect operates on a list of keys.
    # We can create a list of just the sums for bisect, but we need the corresponding mask.
    # So we sort the list of tuples. bisect_left works on tuples if we compare first element?
    # No, bisect compares the whole tuple. We want to compare only the sum.
    # We can use a custom key or just extract sums.
    # Extracting sums is O(N).
    sums_B_values = [s for s, m in sums_B]
    
    best_sum = 0
    best_mask_A = 0
    best_mask_B = 0
    
    # Iterate through sums_A
    # For each s1, we want max s2 such that s1 + s2 < L
    # => s2 < L - s1
    # Let target = L - s1
    # We want the largest value in sums_B_values that is strictly less than target.
    
    for s1, m1 in sums_A:
        target = L - s1
        # Find insertion point for target in sums_B_values
        # bisect_left returns the index where target would be inserted to maintain order.
        # Elements before idx are < target. Elements from idx are >= target.
        # So the candidate is at idx - 1.
        idx = bisect.bisect_left(sums_B_values, target)
        
        if idx > 0:
            s2 = sums_B_values[idx - 1]
            m2 = sums_B[idx - 1][1]
            total_sum = s1 + s2
            
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask_A = m1
                best_mask_B = m2
                
        # Time check inside loop? 8 million iterations in Python is ~1-2 seconds.
        # But if it's slow, we might need to break.
        if time.time() > deadline - 5.0:
            break

    # Construct final mask
    # Mask A is in bits 0..22
    # Mask B is in bits 0..23, shifted by n1
    final_mask = best_mask_A | (best_mask_B << n1)
    
    # Extract indices
    indices = []
    for i in range(n):
        if (final_mask >> i) & 1:
            indices.append(i)
            
    # Validation: Ensure sum is strictly below 1
    # best_sum < L is guaranteed by the search logic (s2 < L - s1 => s1+s2 < L)
    # But let's double check just in case of bugs.
    # Calculate sum of reciprocals for indices
    # sum_frac = sum(Fraction(1, denominators[i]) for i in indices)
    # if sum_frac >= 1:
    #    # Fallback
    #    return _heuristic_fallback(denominators)
    
    return indices

def _heuristic_fallback(denominators):
    # Greedy + Local Search
    n = len(denominators)
    selected = set()
    current_sum = Fraction(0, 1)
    
    # Greedy: sort by reciprocal descending, add if fits
    items = sorted(range(n), key=lambda i: -1/denominators[i])
    
    for i in items:
        if current_sum + Fraction(1, denominators[i]) < 1:
            selected.add(i)
            current_sum += Fraction(1, denominators[i])
            
    # Local Search: try to improve
    # Try swapping one out, one in
    improved = True
    while improved:
        improved = False
        for i in list(selected):
            # Remove i
            new_sum = current_sum - Fraction(1, denominators[i])
            # Try adding j
            for j in range(n):
                if j not in selected:
                    if new_sum + Fraction(1, denominators[j]) < 1:
                        new_total = new_sum + Fraction(1, denominators[j])
                        if new_total > current_sum:
                            selected.remove(i)
                            selected.add(j)
                            current_sum = new_total
                            improved = True
                            break
            if improved:
                break
        # Try adding one item
        if not improved:
            for j in range(n):
                if j not in selected:
                    if current_sum + Fraction(1, denominators[j]) < 1:
                        new_total = current_sum + Fraction(1, denominators[j])
                        if new_total > current_sum:
                            selected.add(j)
                            current_sum = new_total
                            improved = True
                            break
                            
    return list(selected)
