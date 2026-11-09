import time
from math import gcd
from functools import reduce
import bisect
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    denoms = list(range(2, 49))
    n = len(denoms)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    vals = [L // d for d in denoms]
    
    mid = n // 2
    v1 = vals[:mid]
    v2 = vals[mid:]
    
    # Generate sums1
    sums1_vals = array('q', [0])
    sums1_masks = array('I', [0])
    
    for i, v in enumerate(v1):
        current_len = len(sums1_vals)
        # We need to append to the array
        # array doesn't have extend with comprehension easily
        new_vals = array('q')
        new_masks = array('I')
        for k in range(current_len):
            s = sums1_vals[k]
            m = sums1_masks[k]
            ns = s + v
            if ns < L:
                new_vals.append(ns)
                new_masks.append(m | (1 << i))
        
        sums1_vals.extend(new_vals)
        sums1_masks.extend(new_masks)
        
    if time.time() > deadline:
        return _greedy_fallback(denoms)

    # Sort sums1
    # array doesn't have sort. Convert to list of tuples, sort, then back?
    # Or use zip, sort, unzip.
    pairs = list(zip(sums1_vals, sums1_masks))
    pairs.sort(key=lambda x: x[0])
    
    sums1_vals = array('q', [p[0] for p in pairs])
    sums1_masks = array('I', [p[1] for p in pairs])
    
    if time.time() > deadline:
        return _greedy_fallback(denoms)

    # Generate sums2
    sums2_vals = array('q', [0])
    sums2_masks = array('I', [0])
    
    for i, v in enumerate(v2):
        current_len = len(sums2_vals)
        new_vals = array('q')
        new_masks = array('I')
        for k in range(current_len):
            s = sums2_vals[k]
            m = sums2_masks[k]
            ns = s + v
            if ns < L:
                new_vals.append(ns)
                new_masks.append(m | (1 << i))
        
        sums2_vals.extend(new_vals)
        sums2_masks.extend(new_masks)
        
        if time.time() > deadline:
            # If we break here, we have partial sums2
            # We should still try to search with what we have, or fallback?
            # Fallback is safer if time is tight.
            # But let's try to search if we have some sums.
            pass

    if time.time() > deadline:
        return _greedy_fallback(denoms)

    best_total = 0
    best_mask1 = 0
    best_mask2 = 0
    
    # Search
    for k in range(len(sums2_vals)):
        s2 = sums2_vals[k]
        if s2 >= L:
            continue
        limit = L - s2 - 1
        idx = bisect.bisect_right(sums1_vals, limit) - 1
        if idx >= 0:
            s1 = sums1_vals[idx]
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_mask1 = sums1_masks[idx]
                best_mask2 = sums2_masks[k]
                
    if time.time() > deadline:
        return _greedy_fallback(denoms)

    full_mask = best_mask1 | (best_mask2 << mid)
    indices = []
    for i in range(n):
        if full_mask & (1 << i):
            indices.append(i)
            
    if not indices:
        return _greedy_fallback(denoms)
        
    return indices

def _greedy_fallback(denoms):
    from fractions import Fraction
    selected, total = [], Fraction()
    for i, d in enumerate(denoms):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
