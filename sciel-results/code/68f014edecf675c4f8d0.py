import time
import numpy as np
from math import isqrt
from scipy.signal import fftconvolve

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try L = 10^8
    L = 10**8
    
    # Generate powerful numbers up to L
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    
    if time.time() > deadline - 30:
        # Fallback to smaller L
        L = 5 * 10**7
        powerfuls = set()
        max_b = int(L ** (1.0/3.0)) + 2
        for b in range(1, max_b + 1):
            b3 = b * b * b
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerfuls.add(n)
        powerfuls_list = sorted(powerfuls)
    
    max_N = 2 * L
    arr = np.zeros(max_N + 1, dtype=np.float64)
    for p in powerfuls_list:
        arr[p] = 1.0
    
    if time.time() > deadline - 20:
        # Return a small result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    counts = fftconvolve(arr, arr, mode='full')
    
    if time.time() > deadline - 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerfuls_set = set(powerfuls_list)
    best_N = 2
    best_count = 0
    for N in range(2, min(max_N + 1, len(counts))):
        ordered = int(round(counts[N]))
        if ordered == 0:
            continue
        if N % 2 == 0 and (N // 2) in powerfuls_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            best_count = unordered
            best_N = N
    
    # Generate pairs for best_N
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    pairs.sort()
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': pairs}
