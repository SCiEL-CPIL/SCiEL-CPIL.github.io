import time
from math import gcd
from functools import reduce
import numpy as np

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit

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
    
    def local_search_refine(indices, weights, L, start_time, timeout, max_iters=500):
        """Improve solution via swap-based local search with 2-opt style moves."""
        if not indices:
            return indices
        
        from bisect import bisect_left, bisect_right
        
        selected = set(indices)
        current_sum = sum(weights[i] for i in selected)
        best_sum = current_sum
        best_indices = list(indices)
        
        # Precompute unselected items
        all_items = set(range(n))
        unselected = all_items - selected
        
        # Try single swaps: remove one selected, add one unselected
        # And single additions
        for iteration in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            improved = False
            
            # 1. Try single additions (if there's room) - often the easiest improvement
            # Find the largest weight in unselected that fits
            if unselected:
                # Sort unselected by weight descending to find best add quickly
                sorted_unselected = sorted(unselected, key=lambda x: -weights[x])
                for add in sorted_unselected:
                    candidate_sum = current_sum + weights[add]
                    if candidate_sum < L:
                        if candidate_sum > best_sum:
                            selected.add(add)
                            unselected.remove(add)
                            current_sum = candidate_sum
                            best_sum = current_sum
                            best_indices = list(selected)
                            improved = True
                        break  # Since sorted desc, first fit is best for single add
            
            # 2. Try single swaps: remove one selected, add one unselected
            # For each selected item to remove, find best unselected to add
            # We want to maximize (current_sum - w_rem + w_add) < L
            # i.e., w_add < L - current_sum + w_rem
            # To maximize sum, we want largest w_add < L - current_sum + w_rem
            
            if unselected and selected:
                # Precompute sorted unselected for binary search
                unselected_sorted = sorted(unselected, key=lambda x: weights[x])
                unselected_weights_sorted = [weights[x] for x in unselected_sorted]
                
                best_swap_sum = -1
                best_swap_rem = None
                best_swap_add = None
                
                for rem in selected:
                    limit = L - current_sum + weights[rem]
                    # Find largest w_add in unselected_weights_sorted < limit
                    # Use binary search
                    # bisect_left gives index of first element >= limit
                    # We want index - 1
                    idx = bisect_left(unselected_weights_sorted, limit)
                    if idx > 0:
                        add_candidate = unselected_sorted[idx - 1]
                        w_add = weights[add_candidate]
                        candidate_sum = current_sum - weights[rem] + w_add
                        if candidate_sum > best_swap_sum:
                            best_swap_sum = candidate_sum
                            best_swap_rem = rem
                            best_swap_add = add_candidate
                
                if best_swap_rem is not None and best_swap_sum > best_sum:
                    selected.remove(best_swap_rem)
                    selected.add(best_swap_add)
                    unselected.remove(best_swap_add)
                    unselected.add(best_swap_rem)
                    current_sum = best_swap_sum
                    best_sum = current_sum
                    best_indices = list(selected)
                    improved = True
            
            # 3. Try 2-opt swaps: remove two selected, add two unselected
            # This is more complex but can find better solutions
            if len(selected) >= 2 and len(unselected) >= 2:
                # Precompute sorted unselected for binary search
                unselected_sorted = sorted(unselected, key=lambda x: weights[x])
                unselected_weights_sorted = [weights[x] for x in unselected_sorted]
                
                best_2opt_sum = -1
                best_2opt_rem1 = None
                best_2opt_rem2 = None
                best_2opt_add1 = None
                best_2opt_add2 = None
                
                # Try all pairs of selected items to remove
                selected_list = list(selected)
                for i in range(len(selected_list)):
                    for j in range(i + 1, len(selected_list)):
                        rem1 = selected_list[i]
                        rem2 = selected_list[j]
                        w_rem1 = weights[rem1]
                        w_rem2 = weights[rem2]
                        
                        # We need to find two unselected items a1, a2 such that:
                        # current_sum - w_rem1 - w_rem2 + w_a1 + w_a2 < L
                        # i.e., w_a1 + w_a2 < L - current_sum + w_rem1 + w_rem2
                        # To maximize, we want the largest sum of two unselected weights under this limit
                        
                        limit = L - current_sum + w_rem1 + w_rem2
                        
                        # Find best pair of unselected weights that sum to < limit
                        # Use two-pointer on sorted unselected weights
                        left = 0
                        right = len(unselected_weights_sorted) - 1
                        best_pair_sum = -1
                        best_pair_idx1 = -1
                        best_pair_idx2 = -1
                        
                        while left < right:
                            pair_sum = unselected_weights_sorted[left] + unselected_weights_sorted[right]
                            if pair_sum < limit:
                                if pair_sum > best_pair_sum:
                                    best_pair_sum = pair_sum
                                    best_pair_idx1 = left
                                    best_pair_idx2 = right
                                left += 1
                            else:
                                right -= 1
                        
                        if best_pair_sum >= 0:
                            candidate_sum = current_sum - w_rem1 - w_rem2 + best_pair_sum
                            if candidate_sum > best_2opt_sum:
                                best_2opt_sum = candidate_sum
                                best_2opt_rem1 = rem1
                                best_2opt_rem2 = rem2
                                best_2opt_add1 = unselected_sorted[best_pair_idx1]
                                best_2opt_add2 = unselected_sorted[best_pair_idx2]
                
                if best_2opt_rem1 is not None and best_2opt_sum > best_sum:
                    selected.remove(best_2opt_rem1)
                    selected.remove(best_2opt_rem2)
                    selected.add(best_2opt_add1)
                    selected.add(best_2opt_add2)
                    unselected.remove(best_2opt_add1)
                    unselected.remove(best_2opt_add2)
                    unselected.add(best_2opt_rem1)
                    unselected.add(best_2opt_rem2)
                    current_sum = best_2opt_sum
                    best_sum = current_sum
                    best_indices = list(selected)
                    improved = True
            
            if not improved:
                break
        
        return best_indices

    def solve_mitm():
        mid = n // 2  # 23
        left_weights = weights[:mid]
        right_weights = weights[mid:]
        n1 = len(left_weights)
        n2 = len(right_weights)
        
        # Generate subset sums for left half
        # Use array-based generation for speed
        size1 = 1 << n1
        arr1 = [0] * size1
        for i in range(n1):
            w = left_weights[i]
            step = 1 << i
            for j in range(step):
                arr1[step | j] = arr1[j] + w
        
        # Collect valid left subsets (sum < L)
        left_sums = []
        left_masks = []
        for mask in range(size1):
            s = arr1[mask]
            if s < L:
                left_sums.append(s)
                left_masks.append(mask)
        
        if time.time() - start_time > timeout:
            return None
        
        # Generate subset sums for right half
        size2 = 1 << n2
        arr2 = [0] * size2
        for i in range(n2):
            w = right_weights[i]
            step = 1 << i
            for j in range(step):
                arr2[step | j] = arr2[j] + w
        
        # Collect valid right subsets (sum < L)
        right_sums = []
        right_masks = []
        for mask in range(size2):
            s = arr2[mask]
            if s < L:
                right_sums.append(s)
                right_masks.append(mask)
        
        if time.time() - start_time > timeout:
            return None
        
        # Sort right sums for two-pointer
        # Create list of (sum, mask) and sort by sum
        right_pairs = list(zip(right_sums, right_masks))
        right_pairs.sort(key=lambda x: x[0])
        
        # Sort left sums descending for two-pointer
        # Actually, standard two-pointer:
        # Sort A ascending, B ascending.
        # i = 0 (smallest A), j = len(B)-1 (largest B).
        # While i < len(A) and j >= 0:
        #   sum = A[i] + B[j]
        #   if sum < L:
        #      update best
        #      i += 1
        #   else:
        #      j -= 1
        
        left_pairs = list(zip(left_sums, left_masks))
        left_pairs.sort(key=lambda x: x[0])
        
        i = 0
        j = len(right_pairs) - 1
        best_sum = -1
        best_l_mask = 0
        best_r_mask = 0
        
        while i < len(left_pairs) and j >= 0:
            s_l, m_l = left_pairs[i]
            s_r, m_r = right_pairs[j]
            total = s_l + s_r
            
            if total < L:
                if total > best_sum:
                    best_sum = total
                    best_l_mask = m_l
                    best_r_mask = m_r
                i += 1
            else:
                j -= 1
        
        if best_sum > 0:
            indices = []
            for k in range(n1):
                if best_l_mask & (1 << k):
                    indices.append(k)
            for k in range(n2):
                if best_r_mask & (1 << k):
                    indices.append(n1 + k)
            
            # Local search refinement
            indices = local_search_refine(indices, weights, L, start_time, timeout)
            return indices
            
        return None

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
