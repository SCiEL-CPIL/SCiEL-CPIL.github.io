import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    # MITM for N=47 is typically very fast (< 1-2s) in Python if optimized, 
    # but let's keep a generous deadline to be safe against slower environments.
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # L = lcm(2, ..., 48)
    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]
    
    # Split items into two halves
    # 23 items in half1, 24 items in half2
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]
    n1 = len(half1)
    n2 = len(half2)
    
    # --- Generate Feasible Subset Sums for Half 1 ---
    # We store (sum, mask) pairs.
    # Since we need to iterate through all of them to check against half2,
    # we don't need to sort them, but we might filter out sum >= L.
    sums1 = [(0, 0)]
    for i in range(n1):
        w = half1[i]
        current_len = len(sums1)
        for j in range(current_len):
            s, m = sums1[j]
            new_s = s + w
            if new_s < L:
                sums1.append((new_s, m | (1 << i)))
    
    # --- Generate Feasible Subset Sums for Half 2 ---
    # We will deduplicate these to speed up binary search.
    sums2_raw = [(0, 0)]
    for i in range(n2):
        w = half2[i]
        current_len = len(sums2_raw)
        for j in range(current_len):
            s, m = sums2_raw[j]
            new_s = s + w
            if new_s < L:
                sums2_raw.append((new_s, m | (1 << i)))
    
    # Deduplicate sums2: keep the mask for the first occurrence of each sum
    # Using a dictionary for O(1) lookups during dedup
    sum_to_mask2 = {}
    for s, m in sums2_raw:
        if s not in sum_to_mask2:
            sum_to_mask2[s] = m
    
    # Extract sorted unique sums for binary search
    sorted_sums2 = sorted(sum_to_mask2.keys())
    
    # --- Search for Maximum Total Sum < L ---
    best_deficit = L # Initial max deficit
    best_mask1 = 0
    best_mask2 = 0
    
    # Optimization: Sort sums1 by sum descending? 
    # If we find a very small deficit early, we can prune the search?
    # Not easily, because a smaller s1 allows a larger s2.
    # However, if best_deficit is 1, we can break.
    
    # Iterate through all valid subsets of half1
    for s1, m1 in sums1:
        if time.time() > deadline:
            break
            
        # We need s2 such that s1 + s2 < L  =>  s2 < L - s1
        target = L - s1
        
        # Find the largest s2 in sorted_sums2 that is strictly less than target
        # bisect_left returns the index where target would be inserted to keep sorted order.
        # If target is 10 and list is [1, 5, 12], bisect_left(list, 10) returns 2 (index of 12).
        # So the largest element < 10 is at index 2 - 1 = 1 (which is 5).
        idx = bisect_left(sorted_sums2, target)
        
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            total_sum = s1 + s2
            deficit = L - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = m1
                best_mask2 = sum_to_mask2[s2]
                
            # If deficit is 1, it's the minimum possible positive integer deficit.
            if best_deficit == 1:
                break
    
    # Reconstruct indices
    final_indices = []
    
    # Indices from half1 (0 to mid-1)
    for i in range(n1):
        if best_mask1 & (1 << i):
            final_indices.append(i)
            
    # Indices from half2 (mid to n-1)
    for i in range(n2):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)
            
    # Fallback if no valid nonempty subset found (should not happen as 1/2 < 1)
    if not final_indices:
        # Pick the index with the largest single weight (closest to L)
        # This corresponds to the smallest denominator, which is 2 (index 0)
        max_weight_idx = 0 
        final_indices = [max_weight_idx]
        
    return final_indices
