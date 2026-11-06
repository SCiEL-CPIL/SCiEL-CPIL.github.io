import time
from math import lcm
import bisect
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    # ILP Branch and Bound: maximize sum(w_i * x_i) subject to sum(w_i * x_i) <= L-1
    # x_i in {0,1}
    
    # Sort items by weight descending for better pruning
    # But we need to map back to original indices
    items = sorted(range(n), key=lambda i: -weights[i])
    sorted_weights = [weights[i] for i in items]
    
    # Suffix sums for upper bound calculation
    # suffix_sum[k] = sum of sorted_weights[k:]
    suffix_sum = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + sorted_weights[k]
    
    best_sum = 0
    best_mask = 0  # bitmask over sorted items
    
    def branch_and_bound():
        nonlocal best_sum, best_mask, deadline
        
        # State: (index, current_sum, current_mask)
        # We'll use a stack for DFS
        stack = [(0, 0, 0)]
        
        while stack:
            if check_time():
                break
                
            idx, cur_sum, cur_mask = stack.pop()
            
            if idx >= n:
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_mask = cur_mask
                continue
            
            # Option 1: Don't take item idx
            # Check if we can still beat best_sum without this item
            # Upper bound: cur_sum + suffix_sum[idx+1]
            if cur_sum + suffix_sum[idx + 1] > best_sum:
                stack.append((idx + 1, cur_sum, cur_mask))
            
            # Option 2: Take item idx
            new_sum = cur_sum + sorted_weights[idx]
            if new_sum <= L - 1:
                # Upper bound: new_sum + suffix_sum[idx+1]
                if new_sum + suffix_sum[idx + 1] > best_sum:
                    new_mask = cur_mask | (1 << idx)
                    stack.append((idx + 1, new_sum, new_mask))
    
    # Run the solver
    branch_and_bound()
    
    # Convert mask back to original indices
    if best_mask > 0:
        result_indices = []
        for i in range(n):
            if best_mask & (1 << i):
                result_indices.append(items[i])
        
        # Validate
        if result_indices:
            total = sum(weights[i] for i in result_indices)
            if 0 < total < L:
                return result_indices
    
    # Fallback: meet-in-the-middle
    mid = n // 2
    half1_idx = list(range(mid))
    half2_idx = list(range(mid, n))
    
    w1 = [weights[i] for i in half1_idx]
    w2 = [weights[i] for i in half2_idx]

    def gen_sums(weights_list):
        """Generate all subset sums < L"""
        sums = [0]
        for w in weights_list:
            if check_time():
                break
            new_sums = []
            for s in sums:
                ns = s + w
                if ns < L:
                    new_sums.append(ns)
            sums.extend(new_sums)
        return sums

    def find_mask_for_sum(target, weights_list, indices_offset):
        """Find a subset of weights_list that sums to target. Returns list of global indices."""
        if target == 0:
            return []
        
        n_items = len(weights_list)
        pairs = [(0, 0)]
        for i, w in enumerate(weights_list):
            if check_time():
                break
            bit = 1 << i
            current_len = len(pairs)
            for j in range(current_len):
                s, m = pairs[j]
                ns = s + w
                if ns == target:
                    full_mask = m | bit
                    indices = []
                    for k in range(n_items):
                        if full_mask & (1 << k):
                            indices.append(indices_offset + k)
                    return indices
                elif ns < L:
                    pairs.append((ns, m | bit))
        
        return None

    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)
    
    if check_time():
        return [0]

    sums1.sort()
    sums2.sort()
    
    j = len(sums2) - 1
    best_sum_mim = -1
    
    for s1 in sums1:
        while j >= 0 and s1 + sums2[j] >= L:
            j -= 1
            
        if j < 0:
            break
            
        current_sum = s1 + sums2[j]
        if current_sum > best_sum_mim:
            best_sum_mim = current_sum

    if best_sum_mim == -1:
        max_w = max(weights)
        best_sum_mim = max_w
        best_indices = [weights.index(max_w)]
        return best_indices

    set2 = set(sums2)
    target_s1 = None
    target_s2 = None
    
    for s1 in sums1:
        s2_needed = best_sum_mim - s1
        if s2_needed in set2:
            target_s1 = s1
            target_s2 = s2_needed
            break
            
    if target_s1 is None:
        return [0]

    idx1 = find_mask_for_sum(target_s1, w1, 0)
    idx2 = find_mask_for_sum(target_s2, w2, mid)
    
    if idx1 is None or idx2 is None:
        return [0]
        
    final_indices = idx1 + idx2
    
    if not final_indices:
        return [0]
        
    s = sum(weights[i] for i in final_indices)
    if s >= L:
        final_indices.pop()
        s = sum(weights[i] for i in final_indices)
        
    return final_indices
