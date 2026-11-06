import time
from math import lcm
import bisect
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2  # 23
    half1 = list(range(mid))
    half2 = list(range(mid, n))
    
    w1 = [weights[i] for i in half1]
    w2 = [weights[i] for i in half2]
    
    def gen_sums(weights_list):
        """
        Generate all subset sums < L with their masks.
        Returns (sums_list, masks_list) as separate lists for faster iteration.
        """
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_list):
            if time.time() > deadline:
                break
            bit = 1 << i
            current_len = len(sums)
            new_sums = []
            new_masks = []
            L_local = L
            for j in range(current_len):
                ns = sums[j] + w
                if ns < L_local:
                    new_sums.append(ns)
                    new_masks.append(masks[j] | bit)
            sums.extend(new_sums)
            masks.extend(new_masks)
        return sums, masks

    print("Generating sums for half 1...")
    sums1, masks1 = gen_sums(w1)
    print(f"Half 1 size: {len(sums1)}")
    
    if time.time() > deadline:
        return [0] 

    print("Generating sums for half 2...")
    sums2, masks2 = gen_sums(w2)
    print(f"Half 2 size: {len(sums2)}")
    
    if time.time() > deadline:
        return [0]

    # Sort sums2 by sum, keep masks aligned
    order = sorted(range(len(sums2)), key=lambda i: sums2[i])
    sums2_sorted = [sums2[i] for i in order]
    masks2_sorted = [masks2[i] for i in order]
    
    bisect_right = bisect.bisect_right
    L_local = L
    best_sum_local = -1
    best_m1_local = 0
    best_m2_local = 0
    
    # Iterate through sums1
    for idx1 in range(len(sums1)):
        if idx1 % 100000 == 0 and time.time() > deadline:
            break
        s1 = sums1[idx1]
        m1 = masks1[idx1]
        
        target = L_local - 1 - s1
        if target < 0:
            continue
            
        idx = bisect_right(sums2_sorted, target)
        if idx > 0:
            s2 = sums2_sorted[idx - 1]
            total = s1 + s2
            if total > best_sum_local:
                best_sum_local = total
                best_m1_local = m1
                best_m2_local = masks2_sorted[idx - 1]
                
    # Reconstruct indices
    if best_sum_local == -1:
        best_indices = [0]
    else:
        best_indices = []
        for i in range(mid):
            if best_m1_local & (1 << i):
                best_indices.append(i)
        for i in range(mid, n):
            if best_m2_local & (1 << (i - mid)):
                best_indices.append(i)
                
    if not best_indices:
        best_indices = [0]
        
    s = sum(weights[i] for i in best_indices)
    if s >= L:
        best_indices.pop()
        
    print(f"Best sum: {best_sum_local}/{L}")
    print(f"Indices: {best_indices}")
    
    return best_indices
