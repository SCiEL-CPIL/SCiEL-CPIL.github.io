import time
import numpy as np
from math import gcd
from functools import reduce
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        sums = [0]
        for w in weights_half:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return np.array(sums, dtype=np.int64)

    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Use dict to map sum value to one valid coefficient assignment
    def generate_sums_with_coeffs(weights_half):
        # Returns dict: sum_value -> coeffs tuple
        sums_dict = {0: tuple([0] * len(weights_half))}
        for idx, w in enumerate(weights_half):
            new_dict = {}
            for s, coeffs in sums_dict.items():
                for delta in [-w, 0, w]:
                    new_s = s + delta
                    new_coeffs = list(coeffs)
                    if delta == -w:
                        new_coeffs[idx] = -1
                    elif delta == 0:
                        new_coeffs[idx] = 0
                    else:
                        new_coeffs[idx] = 1
                    new_coeffs = tuple(new_coeffs)
                    # Keep first occurrence (arbitrary but consistent)
                    if new_s not in new_dict:
                        new_dict[new_s] = new_coeffs
            sums_dict = new_dict
        return sums_dict

    if time.time() < deadline - 30:
        # Try the dict approach which directly gives us coefficients
        sums1_dict = generate_sums_with_coeffs(w1)
        sums2_dict = generate_sums_with_coeffs(w2)
        
        # Find best nonzero sum
        best_abs = np.iinfo(np.int64).max
        best_s1 = 0
        best_s2 = 0
        best_c1 = None
        best_c2 = None
        
        # Iterate over smaller dict for efficiency
        if len(sums1_dict) <= len(sums2_dict):
            for s1, c1 in sums1_dict.items():
                target = -s1
                if target in sums2_dict:
                    s2 = target
                    total = s1 + s2
                    if total != 0 and abs(total) < best_abs:
                        best_abs = abs(total)
                        best_s1 = s1
                        best_s2 = s2
                        best_c1 = c1
                        best_c2 = sums2_dict[s2]
                
                # Also check nearby values
                # This is expensive, skip for now
            # Check if we found something
            if best_c1 is not None:
                final_coeffs = list(best_c1) + list(best_c2)
                # Verify
                val = sum(final_coeffs[i] / (i + 1) for i in range(n))
                if val != 0:
                    return final_coeffs
        else:
            for s2, c2 in sums2_dict.items():
                target = -s2
                if target in sums1_dict:
                    s1 = target
                    total = s1 + s2
                    if total != 0 and abs(total) < best_abs:
                        best_abs = abs(total)
                        best_s1 = s1
                        best_s2 = s2
                        best_c1 = sums1_dict[s1]
                        best_c2 = c2
            
            if best_c1 is not None:
                final_coeffs = list(best_c1) + list(best_c2)
                val = sum(final_coeffs[i] / (i + 1) for i in range(n))
                if val != 0:
                    return final_coeffs
    
    # Fallback: original meet-in-the-middle with reconstruction
    sums1_sorted = np.sort(sums1)
    targets = -sums2
    
    len_s1 = len(sums1_sorted)
    
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    idx_r = np.searchsorted(sums1_sorted, targets, side='right')
    
    idx_cand1 = idx_l - 1
    valid1 = idx_cand1 >= 0
    idx_cand1_clipped = np.clip(idx_cand1, 0, len_s1 - 1)
    s1_cand1 = sums1_sorted[idx_cand1_clipped]
    
    total1 = sums2 + s1_cand1
    abs_total1 = np.abs(total1)
    
    mask1 = valid1 & (total1 != 0)
    abs_total1[~mask1] = np.iinfo(np.int64).max
    
    idx_cand2 = idx_r
    valid2 = idx_cand2 < len_s1
    idx_cand2_clipped = np.clip(idx_cand2, 0, len_s1 - 1)
    s1_cand2 = sums1_sorted[idx_cand2_clipped]
    
    total2 = sums2 + s1_cand2
    abs_total2 = np.abs(total2)
    
    mask2 = valid2 & (total2 != 0)
    abs_total2[~mask2] = np.iinfo(np.int64).max
    
    min_abs1 = np.min(abs_total1)
    min_abs2 = np.min(abs_total2)
    
    if min_abs1 < min_abs2:
        best_abs = min_abs1
        idx_best_in_1 = np.argmin(abs_total1)
        s2_val = sums2[idx_best_in_1]
        s1_val = s1_cand1[idx_best_in_1]
    else:
        best_abs = min_abs2
        idx_best_in_2 = np.argmin(abs_total2)
        s2_val = sums2[idx_best_in_2]
        s1_val = s1_cand2[idx_best_in_2]
        
    # Reconstruct with memoized DFS
    def reconstruct_coeffs(weights, target_sum):
        n_w = len(weights)
        order = np.argsort(-weights)
        sorted_weights = weights[order]
        
        suffix_max = np.zeros(n_w + 1, dtype=np.int64)
        suffix_min = np.zeros(n_w + 1, dtype=np.int64)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + sorted_weights[i]
            suffix_min[i] = suffix_min[i+1] - sorted_weights[i]
            
        # Memoization: (index, current_sum) -> bool
        memo = {}
        
        def dfs(i, current_sum):
            if time.time() > deadline - 5:
                return None  # Timeout
            if i == n_w:
                return current_sum == target_sum
            
            key = (i, current_sum)
            if key in memo:
                return memo[key]
            
            if current_sum + suffix_max[i] < target_sum:
                memo[key] = False
                return False
            if current_sum + suffix_min[i] > target_sum:
                memo[key] = False
                return False
            
            result = False
            # Try 0 first (most likely to work for small targets)
            if dfs(i + 1, current_sum):
                result = True
            else:
                if dfs(i + 1, current_sum + sorted_weights[i]):
                    result = True
                else:
                    if dfs(i + 1, current_sum - sorted_weights[i]):
                        result = True
            
            memo[key] = result
            return result
        
        def backtrack(i, current_sum, coeffs):
            if time.time() > deadline - 5:
                return False
            if i == n_w:
                return current_sum == target_sum
            
            key = (i, current_sum)
            if key in memo and not memo[key]:
                return False
            
            # Try 0 first
            coeffs[i] = 0
            if backtrack(i + 1, current_sum, coeffs):
                return True
            
            coeffs[i] = 1
            if backtrack(i + 1, current_sum + sorted_weights[i], coeffs):
                return True
            
            coeffs[i] = -1
            if backtrack(i + 1, current_sum - sorted_weights[i], coeffs):
                return True
            
            coeffs[i] = 0
            return False
        
        # First pass: determine feasibility with memo
        feasible = dfs(0, 0)
        if feasible is None:
            # Timeout, try simpler approach
            return [0] * n_w
        if not feasible:
            return [0] * n_w
        
        # Second pass: backtrack to find actual coefficients
        coeffs = [0] * n_w
        if backtrack(0, 0, coeffs):
            orig_coeffs = [0] * n_w
            for i in range(n_w):
                orig_idx = order[i]
                orig_coeffs[orig_idx] = coeffs[i]
            return orig_coeffs
        else:
            return [0] * n_w

    c1 = reconstruct_coeffs(w1, s1_val)
    c2 = reconstruct_coeffs(w2, s2_val)
    
    final_coeffs = c1 + c2
    
    # Verify and fallback if needed
    val = sum(final_coeffs[i] / (i + 1) for i in range(n))
    if val == 0 or any(c not in (-1, 0, 1) for c in final_coeffs):
        # Last resort: simple greedy
        final_coeffs = [0] * n
        # Just pick a single term to ensure nonzero
        final_coeffs[0] = 1
    
    return final_coeffs
