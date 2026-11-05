import time
import sys
from fractions import Fraction
from bisect import bisect_left
from math import log

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers
    weights = [L // d for d in denominators]
    
    # Split into two sets A and B
    # Balance sizes more carefully
    # 24 elements each
    mid = n // 2 
    A_weights = weights[:mid]
    B_weights = weights[mid:]
    
    # Generate subset sums for A
    # Using list of (sum, mask)
    # Optimization: filter out sums >= L early
    sums_A = [(0, 0)]
    for i, w in enumerate(A_weights):
        current_len = len(sums_A)
        new_sums = []
        for k in range(current_len):
            s, m = sums_A[k]
            ns = s + w
            if ns < L:
                new_sums.append((ns, m | (1 << i)))
        # We only keep sums < L because we need total < L
        # Actually, we need to be careful. If A_sum >= L, it can't be part of a valid solution
        # since B_sum >= 0.
        sums_A.extend(new_sums)
        
    # Sort A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # Generate subset sums for B
    sums_B = [(0, 0)]
    for i, w in enumerate(B_weights):
        current_len = len(sums_B)
        new_sums = []
        for k in range(current_len):
            s, m = sums_B[k]
            ns = s + w
            if ns < L:
                new_sums.append((ns, m | (1 << i)))
        sums_B.extend(new_sums)
        
    # Sort B by sum
    sums_B.sort(key=lambda x: x[0])
    
    # Two-pointer technique to find max A_sum + B_sum < L
    # We want to maximize the sum subject to sum < L
    # Iterate i from largest to smallest in A.
    # For each A[i], we want the largest B[j] such that A[i] + B[j] < L.
    
    best_sum = -1
    best_a_idx = 0
    best_b_idx = 0
    
    len_B = len(sums_B)
    # Extract B sums for faster bisect
    B_sums = [x[0] for x in sums_B]
    
    # Iterate A from largest to smallest
    # Note: sums_A is sorted ascending, so we iterate from end to start
    for i in range(len(sums_A) - 1, -1, -1):
        a_sum, a_mask = sums_A[i]
        
        # If a_sum is already too large, we could break?
        # No, smaller a_sum allows larger b_sum.
        
        # Find largest j such that a_sum + B_sums[j] < L
        # => B_sums[j] < L - a_sum
        target = L - a_sum
        
        # bisect_left returns the first index where B_sums[idx] >= target
        # We want the last index where B_sums[idx] < target
        pos = bisect_left(B_sums, target)
        
        if pos > 0:
            candidate_j = pos - 1
            b_sum, b_mask = sums_B[candidate_j]
            total = a_sum + b_sum
            if total > best_sum:
                best_sum = total
                best_a_idx = i
                best_b_idx = candidate_j
        
        # Time check
        if time.time() > deadline - 20:
            break

    if best_sum < 0:
        # Fallback
        return [0, 1, 5]

    # Reconstruct indices
    a_mask = sums_A[best_a_idx][1]
    b_mask = sums_B[best_b_idx][1]
    
    selected = []
    for i in range(mid):
        if a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if b_mask & (1 << i):
            selected.append(mid + i)
            
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1:
        best_result = selected
    else:
        best_result = [0, 1, 5]

    # Polish: Local search to try and improve the deficit
    # Deficit = 1 - sum
    # We want to minimize deficit, i.e., maximize sum.
    # Try swapping elements or adding/removing elements.
    
    current_sum_val = best_sum
    current_mask_full = a_mask | (b_mask << mid)
    
    # Create a set for fast lookups
    current_set = set(selected)
    
    # We can try removing an element and adding two, or swapping one for two, etc.
    # Given time constraints, simple hill climbing:
    # Try adding a single element not in set if sum < L
    # Try removing an element if it helps? No, we want to increase sum.
    # So we look for elements to ADD such that sum + w < L.
    
    # Let's try to add elements that fit.
    # Since we started from the optimal MITM solution, adding any single element might violate sum < L.
    # But maybe we can swap a large element for two smaller ones that sum to more?
    
    # A more effective polish:
    # Iterate through all subsets of size k=2,3,4 from the complement?
    # Or just try single additions first.
    
    # Single addition
    improved = False
    for d_idx in range(n):
        if d_idx not in current_set:
            new_sum = current_sum_val + weights[d_idx]
            if new_sum < L and new_sum > best_sum:
                best_sum = new_sum
                selected.append(d_idx)
                current_set.add(d_idx)
                improved = True
    
    # Now try swaps: remove one, add one?
    # If we remove x and add y, we need sum - w[x] + w[y] < L and > best_sum.
    # This is a bit complex. Let's just try adding more elements greedily.
    
    # Re-run the single addition loop until no improvement
    while True:
        improved = False
        for d_idx in range(n):
            if d_idx not in current_set:
                new_sum = best_sum + weights[d_idx]
                if new_sum < L:
                    best_sum = new_sum
                    selected.append(d_idx)
                    current_set.add(d_idx)
                    improved = True
        if not improved or time.time() > deadline - 5:
            break

    # Verify final result
    if selected:
        total_frac = sum(Fraction(1, denominators[k]) for k in selected)
        if total_frac < 1:
            return selected

    # Fallback
    return [0, 1, 5]
