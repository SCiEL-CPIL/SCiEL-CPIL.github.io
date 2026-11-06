import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
    
    weights = np.array([L // i for i in range(1, N + 1)], dtype=np.int64)
    
    # Split into two halves
    mid = N // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            # sums - w, sums, sums + w
            # This creates 3 arrays of same size, then concatenates
            # Size grows by factor of 3 each step
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    # Generate sums for both halves
    # This might use a lot of memory. 3^15 = 14,348,907.
    # int64 is 8 bytes. 14M * 8 = 112 MB per array.
    # We have two arrays, plus intermediates. ~300-400 MB peak. Should be ok.
    
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums1 for search
    sums1_sorted = np.sort(sums1)
    
    # Check for exact match of sum = 1 or -1
    # We want s1 + s2 = 1 => s1 = 1 - s2
    # We want s1 + s2 = -1 => s1 = -1 - s2
    
    best_abs_num = float('inf')
    best_s1 = None
    best_s2 = None
    
    # Check for +1
    targets_plus = 1 - sums2
    idx_plus = np.searchsorted(sums1_sorted, targets_plus)
    
    # Valid indices are those where idx is within bounds and value matches
    valid_plus = (idx_plus < len(sums1_sorted)) & (idx_plus >= 0)
    # Check match
    matches_plus = valid_plus & (sums1_sorted[idx_plus] == targets_plus)
    
    if np.any(matches_plus):
        # Found a solution with sum 1
        idx_found = np.where(matches_plus)[0][0]
        s2_val = sums2[idx_found]
        s1_val = targets_plus[idx_found]
        best_abs_num = 1
        best_s1 = s1_val
        best_s2 = s2_val
    else:
        # Check for -1
        targets_minus = -1 - sums2
        idx_minus = np.searchsorted(sums1_sorted, targets_minus)
        valid_minus = (idx_minus < len(sums1_sorted)) & (idx_minus >= 0)
        matches_minus = valid_minus & (sums1_sorted[idx_minus] == targets_minus)
        
        if np.any(matches_minus):
            idx_found = np.where(matches_minus)[0][0]
            s2_val = sums2[idx_found]
            s1_val = targets_minus[idx_found]
            best_abs_num = 1
            best_s1 = s1_val
            best_s2 = s2_val
        else:
            # If no exact 1 or -1, find min non-zero difference
            # This is more complex. We can sample or do a more thorough search.
            # But for now, assume 1 or -1 exists.
            # Fallback: use the current program's sparse search if this fails.
            # But let's assume we find 1.
            pass

    if best_abs_num == 1:
        # Reconstruct coefficients
        def reconstruct_coeffs(weights, target_sum):
            n_w = len(weights)
            # Sort by magnitude descending for pruning
            order = np.argsort(-weights)
            sorted_weights = weights[order]
            
            # Suffix sums for pruning
            suffix_max = np.zeros(n_w + 1, dtype=np.int64)
            suffix_min = np.zeros(n_w + 1, dtype=np.int64)
            for i in range(n_w - 1, -1, -1):
                suffix_max[i] = suffix_max[i+1] + sorted_weights[i]
                suffix_min[i] = suffix_min[i+1] - sorted_weights[i]
                
            coeffs = [0] * n_w
            
            def dfs(i, current_sum):
                if i == n_w:
                    return current_sum == target_sum
                
                # Pruning
                if current_sum + suffix_max[i] < target_sum:
                    return False
                if current_sum + suffix_min[i] > target_sum:
                    return False
                    
                # Try -1, 0, 1
                # Order might matter for speed, but 3 options is small
                
                coeffs[i] = -1
                if dfs(i + 1, current_sum - sorted_weights[i]):
                    return True
                    
                coeffs[i] = 0
                if dfs(i + 1, current_sum):
                    return True
                    
                coeffs[i] = 1
                if dfs(i + 1, current_sum + sorted_weights[i]):
                    return True
                    
                return False
            
            if dfs(0, 0):
                orig_coeffs = [0] * n_w
                for i in range(n_w):
                    orig_idx = order[i]
                    orig_coeffs[orig_idx] = coeffs[i]
                return orig_coeffs
            else:
                return None

        c1 = reconstruct_coeffs(w1, best_s1)
        c2 = reconstruct_coeffs(w2, best_s2)
        
        if c1 is not None and c2 is not None:
            final_coeffs = c1 + c2
            # Verify
            val = 0
            for i in range(N):
                if final_coeffs[i] != 0:
                    val += final_coeffs[i] * weights[i]
            if val != 0 and abs(val) == 1:
                return final_coeffs

    # Fallback if MITM failed or didn't find 1
    # Use the sparse search from the current program as a backup
    from itertools import combinations
    
    best_abs_num = float('inf')
    best_coeffs = [1] + [0] * (N - 1)
    
    for k in range(2, 5):
        if time.time() > deadline - 5:
            break
        for combo in combinations(range(N), k):
            w_combo = [weights[i] for i in combo]
            for mask in range(1 << k):
                s = 0
                for j in range(k):
                    if (mask >> j) & 1:
                        s -= w_combo[j]
                    else:
                        s += w_combo[j]
                if s != 0 and abs(s) < best_abs_num:
                    best_abs_num = abs(s)
                    coeffs = [0] * N
                    for j in range(k):
                        coeffs[combo[j]] = -1 if (mask >> j) & 1 else 1
                    best_coeffs = coeffs
                    if best_abs_num == 1:
                        return best_coeffs
                        
    return best_coeffs
