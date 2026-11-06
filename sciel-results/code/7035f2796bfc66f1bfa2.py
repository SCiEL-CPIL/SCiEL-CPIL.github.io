import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def gen_sums(weights):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights):
            current_len = len(sums)
            # List comprehension is faster than loop
            # But we need to filter s + w < L
            # new_sums = []
            # new_masks = []
            # for j in range(current_len):
            #    s = sums[j]
            #    ns = s + w
            #    if ns < L:
            #        new_sums.append(ns)
            #        new_masks.append(masks[j] | (1 << i))
            
            # Using list comprehension with zip
            # We need to iterate over current sums and masks
            new_pairs = []
            bit = 1 << i
            for j in range(current_len):
                ns = sums[j] + w
                if ns < L:
                    new_pairs.append((ns, masks[j] | bit))
            
            if new_pairs:
                sums.extend([p[0] for p in new_pairs])
                masks.extend([p[1] for p in new_pairs])
                
            if time.time() > deadline:
                break
                
        return sums, masks

    sums1, masks1 = gen_sums(w1)
    sums2, masks2 = gen_sums(w2)
    
    # Sort and deduplicate sums2
    # zip, sort, dedup
    zipped2 = sorted(zip(sums2, masks2))
    # Deduplicate: keep first mask for each sum
    unique_sums2 = []
    unique_masks2 = []
    last_sum = -1
    for s, m in zipped2:
        if s != last_sum:
            unique_sums2.append(s)
            unique_masks2.append(m)
            last_sum = s
            
    # Sort sums1 and align masks
    zipped1 = sorted(zip(sums1, masks1))
    unique_sums1 = []
    unique_masks1 = []
    last_sum = -1
    for s, m in zipped1:
        if s != last_sum:
            unique_sums1.append(s)
            unique_masks1.append(m)
            last_sum = s

    # Two pointer
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    i = 0
    j = len(unique_sums2) - 1
    n1 = len(unique_sums1)
    n2 = len(unique_sums2)
    
    while i < n1 and j >= 0:
        if time.time() > deadline:
            break
        s1 = unique_sums1[i]
        s2 = unique_sums2[j]
        total = s1 + s2
        if total >= L:
            j -= 1
        else:
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = unique_masks1[i]
                best_mask2 = unique_masks2[j]
            if best_deficit == 1:
                break
            i += 1
            
    # Reconstruct
    final_indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            final_indices.append(i)
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)
            
    if not final_indices:
        max_weight_idx = max(range(n), key=lambda i: weights[i])
        final_indices = [max_weight_idx]
        
    return final_indices
