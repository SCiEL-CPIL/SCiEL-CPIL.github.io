import time
import numpy as np
from math import gcd
from functools import reduce
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (90s limit)
    
    n = 30
    
    def time_ok():
        return time.time() < deadline
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # weights[i] = L // (i+1), so sum(c[i]*weights[i]) / L = sum(c[i]/(i+1))
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # ===== Phase 1: Meet-in-the-middle to find best candidate =====
    mid = 15
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums
    
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    sums1_sorted = np.sort(sums1)
    targets = -sums2
    
    len_s1 = len(sums1_sorted)
    
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    idx_r = np.searchsorted(sums1_sorted, targets, side='right')
    
    # Predecessor candidates
    idx_cand1 = idx_l - 1
    valid1 = idx_cand1 >= 0
    idx_cand1_clipped = np.clip(idx_cand1, 0, len_s1 - 1)
    s1_cand1 = sums1_sorted[idx_cand1_clipped]
    total1 = sums2 + s1_cand1
    abs_total1 = np.abs(total1)
    mask1 = valid1 & (total1 != 0)
    abs_total1[~mask1] = np.iinfo(np.int64).max
    
    # Successor candidates
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
    
    if min_abs1 <= min_abs2:
        best_abs = min_abs1
        idx_best = np.argmin(abs_total1)
        s2_val = sums2[idx_best]
        s1_val = s1_cand1[idx_best]
    else:
        best_abs = min_abs2
        idx_best = np.argmin(abs_total2)
        s2_val = sums2[idx_best]
        s1_val = s1_cand2[idx_best]
    
    # Reconstruct coefficients via DFS
    def reconstruct_coeffs(weights_arr, target_sum):
        n_w = len(weights_arr)
        order = np.argsort(-weights_arr)
        sorted_weights = weights_arr[order]
        
        suffix_max = np.zeros(n_w + 1, dtype=np.int64)
        suffix_min = np.zeros(n_w + 1, dtype=np.int64)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + sorted_weights[i]
            suffix_min[i] = suffix_min[i+1] - sorted_weights[i]
        
        coeffs = [0] * n_w
        
        def dfs(i, current_sum):
            if i == n_w:
                return current_sum == target_sum
            if current_sum + suffix_max[i] < target_sum:
                return False
            if current_sum + suffix_min[i] > target_sum:
                return False
            
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
        return [0] * n_w
    
    c1 = reconstruct_coeffs(w1, s1_val)
    c2 = reconstruct_coeffs(w2, s2_val)
    
    final_coeffs = np.array(c1 + c2, dtype=np.int64)
    
    # Verify
    current_sum = int(np.sum(final_coeffs * weights))
    if current_sum == 0:
        # Fallback: try a simple non-zero assignment
        final_coeffs = np.zeros(n, dtype=np.int64)
        final_coeffs[0] = 1
        current_sum = int(weights[0])
    
    best_abs = abs(current_sum)
    
    # ===== Phase 2: Local search to improve =====
    if time_ok():
        def compute_sum(coeffs):
            return int(np.dot(coeffs, weights))
        
        def try_flip(coeffs, idx):
            """Try flipping coefficient at idx; return new coeffs and sum if better."""
            old_val = coeffs[idx]
            new_sum = current_sum
            # Try all three values
            best_local = None
            for new_val in (-1, 0, 1):
                if new_val == old_val:
                    continue
                delta = (new_val - old_val) * int(weights[idx])
                candidate_sum = current_sum + delta
                if candidate_sum == 0:
                    continue
                if abs(candidate_sum) < best_abs:
                    best_local = (new_val, candidate_sum, abs(candidate_sum))
            return best_local
        
        # Single-flip local search
        improved = True
        max_iterations = 50000
        iteration = 0
        while improved and time_ok() and iteration < max_iterations:
            improved = False
            iteration += 1
            # Shuffle order to escape local optima
            order = np.random.permutation(n)
            for idx in order:
                if not time_ok():
                    break
                best_local = try_flip(final_coeffs, idx)
                if best_local is not None:
                    new_val, new_sum, new_abs = best_local
                    final_coeffs[idx] = new_val
                    current_sum = new_sum
                    best_abs = new_abs
                    improved = True
                    break  # Restart scan after improvement
        
        # Pair-flip local search
        if time_ok():
            improved = True
            max_pair_iterations = 10000
            pair_iter = 0
            while improved and time_ok() and pair_iter < max_pair_iterations:
                improved = False
                pair_iter += 1
                i1 = np.random.randint(n)
                i2 = np.random.randint(n)
                if i1 == i2:
                    continue
                old1, old2 = final_coeffs[i1], final_coeffs[i2]
                w1v, w2v = int(weights[i1]), int(weights[i2])
                for v1 in (-1, 0, 1):
                    for v2 in (-1, 0, 1):
                        if v1 == old1 and v2 == old2:
                            continue
                        delta = (v1 - old1) * w1v + (v2 - old2) * w2v
                        candidate_sum = current_sum + delta
                        if candidate_sum == 0:
                            continue
                        if abs(candidate_sum) < best_abs:
                            final_coeffs[i1] = v1
                            final_coeffs[i2] = v2
                            current_sum = candidate_sum
                            best_abs = abs(candidate_sum)
                            improved = True
                            break
                    if improved:
                        break
    
    # Final verification
    final_sum = int(np.sum(final_coeffs * weights))
    if final_sum == 0:
        # Ensure non-zero
        final_coeffs[0] = 1
        final_sum = int(weights[0])
    
    return final_coeffs.tolist()
