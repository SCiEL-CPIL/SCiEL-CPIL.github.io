import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n + 1))
    
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate sums for w1
    def gen_sums(w):
        sums = np.array([0], dtype=np.int64)
        for wi in w:
            sums = np.concatenate([sums - wi, sums, sums + wi])
        return sums

    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)
    
    sums2_sorted = np.sort(sums2)
    
    # Vectorized search for min non-zero abs sum
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)
    
    len_s2 = len(sums2_sorted)
    
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    
    abs_r = np.abs(total_r)
    abs_l = np.abs(total_l)
    
    # Mask zeros
    abs_r[total_r == 0] = np.iinfo(np.int64).max
    abs_l[total_l == 0] = np.iinfo(np.int64).max
    
    min_r = np.min(abs_r)
    min_l = np.min(abs_l)
    
    if min_r <= min_l:
        best_abs = min_r
        idx_best = np.argmin(abs_r)
        s1_val = sums1[idx_best]
        s2_val = s2_r[idx_best]
    else:
        best_abs = min_l
        idx_best = np.argmin(abs_l)
        s1_val = sums1[idx_best]
        s2_val = s2_l[idx_best]
        
    # Reconstruct coefficients
    def find_coeffs(w, target):
        # Local MITM for reconstruction
        n_w = len(w)
        mid_r = n_w // 2
        w_r1 = w[:mid_r]
        w_r2 = w[mid_r:]
        
        def gen_with_coeffs(w_part):
            res = [(0, tuple([0]*len(w_part)))]
            for i, wi in enumerate(w_part):
                new_res = []
                for s, c in res:
                    nc1 = list(c); nc1[i] = -1
                    new_res.append((s - wi, tuple(nc1)))
                    nc0 = list(c); nc0[i] = 0
                    new_res.append((s, tuple(nc0)))
                    nc2 = list(c); nc2[i] = 1
                    new_res.append((s + wi, tuple(nc2)))
                res = new_res
            return res
            
        sums_r1 = gen_with_coeffs(w_r1)
        sums_r2 = gen_with_coeffs(w_r2)
        
        map_r2 = {}
        for s, c in sums_r2:
            if s not in map_r2:
                map_r2[s] = c
                
        for s1, c1 in sums_r1:
            s2_needed = target - s1
            if s2_needed in map_r2:
                c2 = map_r2[s2_needed]
                return list(c1) + list(c2)
        return None

    c1 = find_coeffs(w1, s1_val)
    c2 = find_coeffs(w2, s2_val)
    
    final_coeffs = c1 + c2
    
    # Ensure Python ints
    return [int(x) for x in final_coeffs]
