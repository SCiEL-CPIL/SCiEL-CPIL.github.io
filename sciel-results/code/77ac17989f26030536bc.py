import time
import numpy as np
from scipy.signal import fftconvolve
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s buffer for 120s limit
    
    # Generate powerful numbers up to L
    # Increasing L from 10^7 to 10^8 expands search space significantly
    L = 10**8
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    # Create dense array for FFT
    # Size L+1 to allow sums up to 2L
    size = L + 1
    arr = np.zeros(size, dtype=np.int64)
    for p in powerfuls:
        if p < size:
            arr[p] = 1
    
    # Compute convolution to count representations
    # Result length is 2*size - 1 = 2L + 1
    # counts[k] is number of ordered pairs (a,b) such that a+b=k
    counts = fftconvolve(arr, arr, mode='full')
    
    # Find best N
    # We look for max counts in range [2, 2L]
    # Using argmax on a slice is efficient
    best_N = int(np.argmax(counts[2:2*L+1]) + 2)
    
    # Generate pairs for best_N
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls)
    
    pairs = []
    max_a = best_N // 2
    
    for a in powerfuls_list:
        if a > max_a:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
            
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
