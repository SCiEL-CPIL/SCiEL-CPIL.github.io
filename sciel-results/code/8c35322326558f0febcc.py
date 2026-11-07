import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0
    MAX_N = 10_000_000
    
    # Sieve for sigma using NumPy
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(1, MAX_N + 1):
        if time.time() - start_time > deadline - 15:
            break
        sigma[i::i] += i
    
    if time.time() - start_time > deadline - 10:
        return [11, 6]
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[int(sigma[n])].append(n)
    
    if time.time() - start_time > deadline - 5:
        return [11, 6]
    
    best_error = float('inf')
    best_pair = (11, 6)
    
    for s, lst in groups.items():
        if time.time() - start_time > deadline - 3:
            break
        if len(lst) < 2:
            continue
        lst.sort()
        for b in lst:
            target_a = 1.5 * b
            # Binary search
            lo, hi = 0, len(lst) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if lst[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid - 1
            for a_idx in (lo, lo - 1):
                if 0 <= a_idx < len(lst):
                    a = lst[a_idx]
                    if a == b:
                        continue
                    err_num = abs(2 * a - 3 * b)
                    if err_num == 0:
                        continue
                    err = err_num / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    a, b = best_pair
    if 2 * a != 3 * b and 1 <= a <= MAX_N and 1 <= b <= MAX_N:
        return [int(a), int(b)]
    return [11, 6]
