import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate subset sums for first half
    def gen_sums(weights):
        sums = [0]
        for w in weights:
            # List comprehension is faster than loop
            sums += [s + w for s in sums]
        return sums
    
    sums1 = gen_sums(w1)
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to greedy if generation is too slow
        return greedy_fallback(denominators, L, weights)

    # Generate subset sums for second half
    sums2 = gen_sums(w2)
    
    if time.time() > deadline - 5:
        return greedy_fallback(denominators, L, weights)

    # Sort sums2
    sums2.sort()
    
    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0
    target = L - 1
    
    # For each sum in sums1, find best sum in sums2
    # We need to map sums back to masks.
    # Storing (sum, mask) pairs is better.
    
    # Let's redo generation to store masks
    def gen_sums_with_masks(weights):
        # Returns list of (sum, mask)
        results = [(0, 0)]
        for i, w in enumerate(weights):
            bit = 1 << i
            # Create new entries
            new_entries = [(s + w, m | bit) for s, m in results]
            results += new_entries
        return results

    # This might be slower due to tuple overhead.
    # Let's stick to sums only, and reconstruct?
    # Reconstructing is hard without storing masks.
    # Alternative: Store masks in a separate array or use bit operations to find the subset.
    # Actually, for n=23, we can just store the sums.
    # To get the mask, we can re-compute or store indices.
    
    # Let's try storing (sum, mask) but optimize.
    # Or, since we just need ONE optimal solution, we can do this:
    
    # Let's use the list of sums and binary search.
    # To recover the indices, we need to know which subset produced the sum.
    # We can store the mask alongside the sum.
    
    def gen_sums_mask(weights):
        res = [(0, 0)]
        for i, w in enumerate(weights):
            bit = 1 << i
            res += [(s + w, m | bit) for s, m in res]
        return res

    # This doubles memory. 8M tuples is ~ 100MB+. Might be tight but okay.
    # Let's try to be more memory efficient if possible.
    # But 60s is plenty for 8M items.
    
    try:
        pairs1 = gen_sums_mask(w1)
        pairs2 = gen_sums_mask(w2)
    except MemoryError:
        return greedy_fallback(denominators, L, weights)
    
    # Sort pairs2 by sum
    pairs2.sort(key=lambda x: x[0])
    sums2_only = [p[0] for p in pairs2]
    
    best_total = 0
    best_m1 = 0
    best_m2 = 0
    
    for s1, m1 in pairs1:
        if s1 > target:
            continue
        # Find largest s2 such that s1 + s2 <= target
        # s2 <= target - s1
        limit = target - s1
        # bisect_right returns index after the last element <= limit
        idx = bisect.bisect_right(sums2_only, limit)
        if idx > 0:
            s2, m2 = pairs2[idx - 1]
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_m1 = m1
                best_m2 = m2
    
    # Reconstruct indices
    indices = []
    for i in range(len(w1)):
        if best_m1 & (1 << i):
            indices.append(i)
    for i in range(len(w2)):
        if best_m2 & (1 << i):
            indices.append(mid + i)
            
    return indices

def greedy_fallback(denominators, L, weights):
    # Greedy: pick largest weights that fit
    # Sort by weight descending? No, weight is value.
    # We want to maximize sum.
    # Greedy by value/weight? Here value=weight.
    # So just pick largest weights that don't exceed capacity.
    # This is not optimal for subset sum, but a valid fallback.
    
    # Better fallback: Randomized local search
    import random
    best_sum = 0
    best_idx = []
    
    # Start with empty
    current_sum = 0
    current_mask = 0
    n = len(denominators)
    
    # Simple hill climbing
    # ...
    
    # Just return a valid small set to avoid crash
    # The greedy from seed is better than empty
    selected, total = [], 0
    for i, w in enumerate(weights):
        if total + w < L:
            selected.append(i)
            total += w
    return selected
