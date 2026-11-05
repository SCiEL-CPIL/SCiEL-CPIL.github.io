import time
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using the form a^2 * b^3."""
    powerful = set()
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    t0 = time.time()
    deadline = 115.0
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Try different limits
    limits_to_try = [10**6, 3*10**6, 10**7, 3*10**7, 10**8]
    
    for M in limits_to_try:
        if time.time() - t0 > deadline - 25:
            break
        
        powerful = _generate_powerful_numbers(M)
        K = len(powerful)
        powerful_set = set(powerful)
        
        if K < 2:
            continue
        
        # FFT approach
        arr_size = 1
        while arr_size < 2 * M + 1:
            arr_size *= 2
        
        f = np.zeros(arr_size, dtype=np.float64)
        for p in powerful:
            f[p] = 1.0
        
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=arr_size)
        
        # Find top candidates
        max_N_search = min(2 * M, 10**18)
        
        # Get top 2000 indices by conv value
        top_indices = np.argsort(conv)[-2000:]
        top_indices = top_indices[(top_indices >= 2) & (top_indices <= max_N_search)]
        
        if len(top_indices) == 0:
            continue
        
        for idx in top_indices:
            n = int(idx)
            if n < 2 or n > 10**18:
                continue
            
            # Count representations with a <= b
            count = 0
            pairs = []
            for a in powerful:
                if a > n - a:
                    break
                b = n - a
                if b in powerful_set:
                    count += 1
                    pairs.append([a, b])
            
            if count > best_count:
                best_count = count
                best_N = n
                best_pairs = pairs
        
        if time.time() - t0 > deadline - 15:
            break
    
    # Final verification
    verified_pairs = []
    powerful_set = set(_generate_powerful_numbers(best_N))
    for a, b in best_pairs:
        if a + b == best_N and a <= b and a in powerful_set and b in powerful_set:
            verified_pairs.append([a, b])
    
    if not verified_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': verified_pairs}
