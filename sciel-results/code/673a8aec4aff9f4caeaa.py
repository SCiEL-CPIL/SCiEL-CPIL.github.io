import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_efficient(weights_part):
        sums = [0]
        for w in weights_part:
            new_sums = []
            # Pre-allocate size to be slightly faster?
            # len(sums) * 3
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    # Fallback if MIM takes too long to generate
    if time.time() > deadline - 10:
        return [1, -1] + [0] * 28

    sums1 = generate_sums_efficient(w1)
    
    if time.time() > deadline - 10:
        return [1, -1] + [0] * 28

    sums2 = generate_sums_efficient(w2)
    
    if time.time() > deadline - 5:
        return [1, -1] + [0] * 28

    sums1_arr = np.array(sums1, dtype=np.int64)
    sums2_arr = np.array(sums2, dtype=np.int64)
    sums2_sorted = np.sort(sums2_arr)
    
    targets = -sums1_arr
    idx = np.searchsorted(sums2_sorted, targets)
    
    # Candidate 1: idx
    idx_c1 = np.clip(idx, 0, len(sums2_sorted) - 1)
    s2_c1 = sums2_sorted[idx_c1]
    sum_c1 = sums1_arr + s2_c1
    abs_c1 = np.abs(sum_c1)
    abs_c1[sum_c1 == 0] = np.iinfo(np.int64).max
    
    # Candidate 2: idx - 1
    idx_c2 = np.clip(idx - 1, 0, len(sums2_sorted) - 1)
    s2_c2 = sums2_sorted[idx_c2]
    sum_c2 = sums1_arr + s2_c2
    abs_c2 = np.abs(sum_c2)
    abs_c2[sum_c2 == 0] = np.iinfo(np.int64).max
    
    min1 = np.min(abs_c1)
    min2 = np.min(abs_c2)
    
    if min1 <= min2:
        best_abs = min1
        idx_best = np.argmin(abs_c1)
        best_s1_val = sums1_arr[idx_best]
        best_s2_val = s2_c1[idx_best]
    else:
        best_abs = min2
        idx_best = np.argmin(abs_c2)
        best_s1_val = sums1_arr[idx_best]
        best_s2_val = s2_c2[idx_best]
        
    if best_abs == np.iinfo(np.int64).max:
        return [1, -1] + [0] * 28

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

    c1 = find_coefficients(w1, best_s1_val)
    c2 = find_coefficients(w2, best_s2_val)
    
    if c1 is None or c2 is None:
        return [1, -1] + [0] * 28

    coeffs = c1 + c2
    return coeffs
