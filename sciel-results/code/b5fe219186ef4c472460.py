import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    def generate_sums_numpy(weights_part):
        sums = np.array([0], dtype=np.int64)
        for w in weights_part:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums
    
    def find_coefficients(weights_part, target_sum):
        n_part = len(weights_part)
        coeffs = [0] * n_part
        def dfs(idx, current_sum):
            if idx == n_part:
                return current_sum == target_sum
            w = weights_part[idx]
            for c in [-1, 0, 1]:
                coeffs[idx] = c
                if dfs(idx + 1, current_sum + c * w):
                    return True
            return False
        if dfs(0, 0):
            return coeffs
        return None
    
    def local_search(coeffs):
        """Try to improve the solution by flipping individual coefficients."""
        if time.time() > deadline - 3:
            return coeffs
        
        best_coeffs = list(coeffs)
        best_abs = abs(sum(c * w for c, w in zip(best_coeffs, weights)))
        
        for _ in range(100):  # Limit iterations
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(n):
                for new_c in [-1, 0, 1]:
                    if new_c == best_coeffs[i]:
                        continue
                    old_c = best_coeffs[i]
                    best_coeffs[i] = new_c
                    new_abs = abs(sum(c * w for c, w in zip(best_coeffs, weights)))
                    if new_abs < best_abs and new_abs != 0:
                        best_abs = new_abs
                        improved = True
                    else:
                        best_coeffs[i] = old_c
            if not improved:
                break
        
        return best_coeffs
    
    def try_split(mid):
        if time.time() > deadline - 8:
            return None
        
        w1 = weights[:mid]
        w2 = weights[mid:]
        
        sums1 = generate_sums_numpy(w1)
        if time.time() > deadline - 5:
            return None
        
        sums2 = generate_sums_numpy(w2)
        if time.time() > deadline - 3:
            return None
        
        sums2_sorted = np.sort(sums2)
        
        targets = -sums1
        idx = np.searchsorted(sums2_sorted, targets)
        
        idx_c1 = np.clip(idx, 0, len(sums2_sorted) - 1)
        s2_c1 = sums2_sorted[idx_c1]
        total_c1 = sums1 + s2_c1
        abs_c1 = np.abs(total_c1)
        abs_c1[total_c1 == 0] = np.iinfo(np.int64).max
        
        idx_c2 = np.clip(idx - 1, 0, len(sums2_sorted) - 1)
        s2_c2 = sums2_sorted[idx_c2]
        total_c2 = sums1 + s2_c2
        abs_c2 = np.abs(total_c2)
        abs_c2[total_c2 == 0] = np.iinfo(np.int64).max
        
        min1 = np.min(abs_c1)
        min2 = np.min(abs_c2)
        
        if min1 <= min2:
            best_abs = min1
            idx_best = np.argmin(abs_c1)
            best_s1_val = sums1[idx_best]
            best_s2_val = s2_c1[idx_best]
        else:
            best_abs = min2
            idx_best = np.argmin(abs_c2)
            best_s1_val = sums1[idx_best]
            best_s2_val = s2_c2[idx_best]
        
        if best_abs == np.iinfo(np.int64).max:
            return None
        
        c1 = find_coefficients(w1, best_s1_val)
        c2 = find_coefficients(w2, best_s2_val)
        
        if c1 is None or c2 is None:
            return None
        
        coeffs = c1 + c2
        return (best_abs, coeffs)
    
    # Try 15/15 split first
    best_result = try_split(15)
    
    # Try other splits if time permits
    if best_result is None or time.time() < deadline - 15:
        result2 = try_split(12)
        if result2 is not None and (best_result is None or result2[0] < best_result[0]):
            best_result = result2
    
    if best_result is None or time.time() < deadline - 8:
        result3 = try_split(18)
        if result3 is not None and (best_result is None or result3[0] < best_result[0]):
            best_result = result3
    
    if best_result is not None:
        coeffs = best_result[1]
        # Local search to improve
        coeffs = local_search(coeffs)
        return coeffs
    
    # Fallback
    return [1, -1] + [0] * 28
