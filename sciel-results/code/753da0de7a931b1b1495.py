import time
import numpy as np
from math import isqrt

def _is_squarefree(n):
    """Check if n is squarefree."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    result = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        if not _is_squarefree(b):
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                result.add(val)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Try increasing limits until we find one that fits in memory/time
    # Start with a high limit to maximize score
    limits_to_try = [2 * 10**8, 10**8, 5 * 10**7]
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    for L in limits_to_try:
        if time.time() > deadline - 30:
            break
            
        try:
            # Generate powerful numbers
            powerful_set = _generate_powerful(L)
            if not powerful_set:
                continue
                
            # Determine FFT size
            # We need to find N up to 2*L
            # FFT size must be >= 2*L
            # Use next power of 2 for efficiency
            n_fft = 1
            while n_fft < 2 * L:
                n_fft *= 2
                
            # Create signal
            # Use float64 for precision in FFT
            A = np.zeros(n_fft, dtype=np.float64)
            for p in powerful_set:
                if p < n_fft:
                    A[p] = 1.0
            
            # FFT
            F = np.fft.rfft(A)
            del A
            
            # Autocorrelation (squared in freq domain)
            G = F * F
            del F
            
            # Inverse FFT
            C = np.fft.irfft(G, n=n_fft)
            del G
            
            # Find max count for N in [2, 2*L]
            # C[0] is sum 0, C[1] is sum 1, etc.
            max_search_range = min(2 * L, n_fft - 1)
            if max_search_range < 2:
                continue
                
            # Slice to valid range
            valid_C = C[2:max_search_range+1]
            if len(valid_C) == 0:
                continue
                
            max_idx_relative = np.argmax(valid_C)
            N_best = max_idx_relative + 2
            count_best = int(valid_C[max_idx_relative])
            
            if count_best > best_count:
                # Extract pairs
                pairs = []
                # To ensure a <= b, we can just check all a in set
                # and if b = N - a is in set and a <= b
                for a in powerful_set:
                    b = N_best - a
                    if b in powerful_set and a <= b:
                        pairs.append([a, b])
                
                # Sort pairs for consistency (optional but good)
                pairs.sort()
                
                # Verify count matches
                if len(pairs) == count_best:
                    best_count = count_best
                    best_result = {'N': N_best, 'pairs': pairs}
                else:
                    # If mismatch due to floating point error in argmax,
                    # we might have picked a slightly wrong peak.
                    # But usually argmax is robust.
                    # If mismatch, we can still return the pairs we found,
                    # but the score is len(pairs).
                    # Let's update best if len(pairs) is better.
                    if len(pairs) > best_count:
                         best_count = len(pairs)
                         best_result = {'N': N_best, 'pairs': pairs}

        except MemoryError:
            continue
        except Exception as e:
            # If something goes wrong, try next smaller limit
            continue

    return best_result
