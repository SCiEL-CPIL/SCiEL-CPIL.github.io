import time
import math
import bisect
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 82  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Fast sigma computation using smallest prime factor sieve with NumPy
    spf = np.arange(MAX_N + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == spf[i*i::i], i, spf[i*i::i])
    
    # Compute sigma using the multiplicative property
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # Convert to list for the iterative sigma computation (faster in Python for this pattern)
    spf_list = spf.tolist()
    sigma_list = [0] * (MAX_N + 1)
    sigma_list[1] = 1
    for n in range(2, MAX_N + 1):
        p = spf_list[n]
        m = n // p
        power = p
        while m > 1 and m % p == 0:
            power *= p
            m //= p
        factor = (power * p - 1) // (p - 1)
        if m == 1:
            sigma_list[n] = factor
        else:
            sigma_list[n] = sigma_list[m] * factor
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    # Group indices by sigma value
    groups = {}
    for i in range(1, MAX_N + 1):
        s = sigma_list[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    best_error = float('inf')
    best_a = None
    best_b = None
    
    # Process each group
    for s, lst in groups.items():
        if time.time() - start_time > DEADLINE:
            break
            
        if len(lst) < 2:
            continue
            
        # lst is already sorted because we appended in increasing order of i
        
        # Two-pointer approach: for each b, find a close to 1.5*b
        ptr_a = 0
        n_lst = len(lst)
        
        for b in lst:
            # Move ptr_a to be close to 1.5 * b
            target_a = 1.5 * b
            while ptr_a < n_lst and lst[ptr_a] < target_a:
                ptr_a += 1
                
            # Check a window around ptr_a for best candidate
            # Check positions ptr_a-2 through ptr_a+2 to handle gaps
            lo = max(0, ptr_a - 2)
            hi = min(n_lst, ptr_a + 3)
            for idx in range(lo, hi):
                a = lst[idx]
                if a == b:
                    continue
                err_num = abs(2*a - 3*b)
                if err_num == 0:
                    continue
                err = err_num / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                        
    if best_a is None:
        return [11, 6]
        
    # Final polish: do a targeted search for pairs with very close ratios
    # We know the best error so far. Let's try to find even better pairs
    # by focusing on groups with larger sizes (more candidates)
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Sort groups by size descending to process larger groups first
        # (larger groups have more chances of close pairs)
        large_groups = [(s, lst) for s, lst in groups.items() if len(lst) >= 3]
        large_groups.sort(key=lambda x: -len(x[1]))
        
        for s, lst in large_groups:
            if time.time() - start_time > DEADLINE:
                break
                
            n_lst = len(lst)
            if n_lst < 2:
                continue
            
            # For this group, do a more thorough search
            # For each b, use bisect to find the closest a to 1.5*b
            for i_b, b in enumerate(lst):
                target_a = 1.5 * b
                pos = bisect.bisect_left(lst, target_a)
                
                # Check a few positions around pos
                lo = max(0, pos - 3)
                hi = min(n_lst, pos + 4)
                for idx in range(lo, hi):
                    a = lst[idx]
                    if a == b:
                        continue
                    err_num = abs(2*a - 3*b)
                    if err_num == 0:
                        continue
                    err = err_num / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
