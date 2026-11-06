import time
from math import gcd
from functools import reduce
from bisect import bisect_right
import numpy as np

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit

    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    def solve_mitm():
        mid = n // 2  # 23
        left_weights = weights[:mid]
        right_weights = weights[mid:]
        
        def gen_subsets_fast(w_list):
            """Generate subset sums using iterative expansion with pruning."""
            # Start with sum=0, mask=0
            sums = [0]
            masks = [0]
            for i, w in enumerate(w_list):
                bit = 1 << i
                current_len = len(sums)
                # Only add new subsets where sum + w < L
                new_sums = []
                new_masks = []
                for k in range(current_len):
                    ns = sums[k] + w
                    if ns < L:
                        new_sums.append(ns)
                        new_masks.append(masks[k] | bit)
                sums.extend(new_sums)
                masks.extend(new_masks)
            # Return list of (sum, mask) tuples
            return list(zip(sums, masks))

        # Generate left subsets
        left_subsets = gen_subsets_fast(left_weights)
        left_subsets.sort(key=lambda x: x[0])
        
        if time.time() - start_time > timeout:
            return None

        # Generate right subsets
        right_subsets = gen_subsets_fast(right_weights)
        right_subsets.sort(key=lambda x: x[0])
        
        if time.time() - start_time > timeout:
            return None

        # Two-pointer to find max sum < L
        i = len(left_subsets) - 1
        j = 0
        best_sum = 0
        best_l_mask = 0
        best_r_mask = 0
        
        while i >= 0 and j < len(right_subsets):
            s_l = left_subsets[i][0]
            s_r = right_subsets[j][0]
            total = s_l + s_r
            if total < L:
                if total > best_sum:
                    best_sum = total
                    best_l_mask = left_subsets[i][1]
                    best_r_mask = right_subsets[j][1]
                j += 1
            else:
                i -= 1
                
        if best_sum > 0:
            indices = []
            for k in range(mid):
                if best_l_mask & (1 << k):
                    indices.append(k)
            for k in range(len(right_weights)):
                if best_r_mask & (1 << k):
                    indices.append(mid + k)
                    
            # Local search refinement
            indices = local_search_refine(indices, weights, L, start_time, timeout)
            return indices
            
        return None

    def local_search_refine(indices, weights, L, start_time, timeout):
        """Improve solution via swap-based local search."""
        if not indices:
            return indices
        
        selected = set(indices)
        current_sum = sum(weights[i] for i in selected)
        best_sum = current_sum
        best_indices = list(indices)
        
        # Precompute unselected items
        all_items = set(range(n))
        unselected = all_items - selected
        
        # Try single swaps: remove one selected, add one unselected
        for _ in range(50):  # Limit iterations
            if time.time() - start_time > timeout:
                break
            
            improved = False
            # Try removing each selected item and adding best unselected
            for rem in list(selected):
                new_sum_without = current_sum - weights[rem]
                if new_sum_without >= L:
                    continue
                # Find best unselected item to add
                best_add = None
                best_add_sum = new_sum_without
                for add in unselected:
                    candidate_sum = new_sum_without + weights[add]
                    if candidate_sum < L and candidate_sum > best_add_sum:
                        best_add_sum = candidate_sum
                        best_add = add
                
                if best_add is not None and best_add_sum > best_sum:
                    # Apply swap
                    selected.remove(rem)
                    selected.add(best_add)
                    unselected.remove(best_add)
                    unselected.add(rem)
                    current_sum = best_add_sum
                    best_sum = current_sum
                    best_indices = list(selected)
                    improved = True
                    break
            
            if not improved:
                # Try single additions (if there's room)
                best_add = None
                best_add_sum = current_sum
                for add in unselected:
                    candidate_sum = current_sum + weights[add]
                    if candidate_sum < L and candidate_sum > best_add_sum:
                        best_add_sum = candidate_sum
                        best_add = add
                
                if best_add is not None:
                    selected.add(best_add)
                    unselected.remove(best_add)
                    current_sum = best_add_sum
                    best_sum = current_sum
                    best_indices = list(selected)
                    improved = True
            
            if not improved:
                break
        
        return best_indices

    result = solve_mitm()
    if result:
        return result
        
    # Fallback: greedy + local search
    # Sort by weight descending, greedily add if sum < L
    order = sorted(range(n), key=lambda i: -weights[i])
    selected = []
    current_sum = 0
    for idx in order:
        if current_sum + weights[idx] < L:
            selected.append(idx)
            current_sum += weights[idx]
    
    if not selected:
        return [0]
    
    # Refine with local search
    selected = local_search_refine(selected, weights, L, start_time, timeout)
    return selected
