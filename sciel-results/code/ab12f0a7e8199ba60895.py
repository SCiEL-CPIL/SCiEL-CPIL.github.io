import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 30
    
    # 1. Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # 2. Compute weights w_i = L // i
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i
        
    mid = n // 2
    # Indices 1..mid for left, mid+1..n for right
    left_weights = [w[i] for i in range(1, mid + 1)]
    right_weights = [w[i] for i in range(mid + 1, n + 1)]
    
    def generate_sums_masks(weights):
        """
        Generates all possible signed sums and their base-3 mask encodings.
        Returns (sums_array, masks_array)
        """
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        
        for wt in weights:
            # d=0 -> c=-1, delta=-wt, mask*3
            s0 = sums - wt
            m0 = masks * 3
            
            # d=1 -> c=0, delta=0, mask*3+1
            s1 = sums
            m1 = masks * 3 + 1
            
            # d=2 -> c=1, delta=wt, mask*3+2
            s2 = sums + wt
            m2 = masks * 3 + 2
            
            sums = np.concatenate([s0, s1, s2])
            masks = np.concatenate([m0, m1, m2])
            
            if time.time() - start_time > deadline:
                return None, None
                
        return sums, masks

    # Generate left half sums and masks
    left_sums, left_masks = generate_sums_masks(left_weights)
    if left_sums is None:
        return [1] + [0]*29
        
    # Sort left sums and align masks
    idx = np.argsort(left_sums)
    left_sums = left_sums[idx]
    left_masks = left_masks[idx]
    
    # Generate right half sums and masks
    right_sums, right_masks = generate_sums_masks(right_weights)
    if right_sums is None:
        return [1] + [0]*29

    # Sort right sums and align masks
    idx = np.argsort(right_sums)
    right_sums = right_sums[idx]
    right_masks = right_masks[idx]
    
    # Convert to lists for faster iteration in Python loop
    # This trade-off (memory for speed) is worth it for 14M elements
    ls = left_sums.tolist()
    lm = left_masks.tolist()
    rs = right_sums.tolist()
    rm = right_masks.tolist()
    
    n_left = len(ls)
    n_right = len(rs)
    
    i = 0
    j = n_right - 1
    
    best_abs = None
    best_l_mask = None
    best_r_mask = None
    
    # Two-pointer sweep to find minimum non-zero absolute sum
    while i < n_left and j >= 0:
        s_val = ls[i] + rs[j]
        
        if s_val != 0:
            abs_val = abs(s_val)
            if best_abs is None or abs_val < best_abs:
                best_abs = abs_val
                best_l_mask = lm[i]
                best_r_mask = rm[j]
                if best_abs == 1:
                    break
        else:
            # If sum is 0, we ignore it for the "best" but check neighbors
            # to ensure we don't miss the best non-zero if the global min is 0
            # Check (i, j-1)
            if j > 0:
                s_check = ls[i] + rs[j-1]
                if s_check != 0:
                    abs_check = abs(s_check)
                    if best_abs is None or abs_check < best_abs:
                        best_abs = abs_check
                        best_l_mask = lm[i]
                        best_r_mask = rm[j-1]
            # Check (i+1, j)
            if i < n_left - 1:
                s_check = ls[i+1] + rs[j]
                if s_check != 0:
                    abs_check = abs(s_check)
                    if best_abs is None or abs_check < best_abs:
                        best_abs = abs_check
                        best_l_mask = lm[i+1]
                        best_r_mask = rm[j]
                        
            if best_abs == 1:
                break

        # Move pointers
        if s_val > 0:
            j -= 1
        elif s_val < 0:
            i += 1
        else:
            # If sum is 0, move both to progress
            i += 1
            j -= 1
            
        if time.time() - start_time > deadline:
            break

    if best_l_mask is None:
        # Fallback if no non-zero sum found (should not happen)
        return [1] + [0]*29
        
    # 5. Decode coefficients
    coeffs = [0] * n
    
    # Decode left mask for indices 0..mid-1
    d = best_l_mask
    for k in range(mid):
        digit = d % 3
        d //= 3
        c = digit - 1 # 0->-1, 1->0, 2->1
        coeffs[k] = c
        
    # Decode right mask for indices mid..n-1
    d = best_r_mask
    for k in range(n - mid):
        digit = d % 3
        d //= 3
        c = digit - 1
        coeffs[mid + k] = c
        
    return coeffs
