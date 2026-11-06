import time
import numpy as np
from math import isqrt
from collections import defaultdict

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2).
    1 counts as powerful."""
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1
    if x > 1:
        return False
    return True

def _generate_powerful(L, primes):
    """Generate all powerful numbers <= L using recursive prime-power products.
    A powerful number is prod p_i^{e_i} with e_i >= 2."""
    powerful = [1]
    for p in primes:
        p2 = p * p
        if p2 > L:
            break
        # For existing powerful numbers, multiply by p^2, p^3, ...
        new_adds = []
        for pw in powerful:
            val = pw * p2
            while val <= L:
                new_adds.append(val)
                val *= p
        powerful.extend(new_adds)
    return sorted(set(powerful))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Generate small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate powerful numbers up to a moderate limit
    # Use recursive generation with small primes
    L = 20 * 10**6
    powerful = _generate_powerful(L, small_primes)
    # Filter to ensure correctness
    powerful = [n for n in powerful if n <= L and _is_powerful(n)]
    powerful_set = set(powerful)
    
    elapsed = time.time() - start_time
    print(f"Generated {len(powerful)} powerful numbers up to {L}, took {elapsed:.1f}s")
    
    if time.time() > deadline - 10:
        # Fallback: return a known good result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Find N that maximizes pairs.
    # Key insight: if we pick N = k * M where M is a large powerful number,
    # then N - a is powerful for many small powerful a's if N is chosen well.
    # 
    # Better: use the fact that small powerful numbers are dense near 0.
    # The number of powerful numbers up to X is ~ c * sqrt(X).
    # So for N, the number of pairs is roughly the number of powerful a <= N/2
    # such that N-a is also powerful.
    #
    # Let's try candidate N values that are sums of many small powerful numbers.
    # Good candidates: N = 2 * (sum of first k powerful numbers / k) approximately,
    # or N values that are multiples of small powerful numbers.
    
    # Approach: For each powerful number a in the list, consider N = a + b where b is also powerful.
    # To find the N with max representations, we can:
    # 1. Try N values that are "central" - around the median of powerful number sums
    # 2. Or use a smarter method: count for candidate N values
    
    # Let's use a hybrid: generate candidate N values from sums of pairs of small powerful numbers,
    # then count representations for the most promising candidates.
    
    # First, let's try a direct approach with a moderate-sized powerful list
    # Use FFT convolution for efficiency
    n_pwr = len(powerful)
    print(f"Number of powerful numbers: {n_pwr}")
    
    if n_pwr < 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    max_val = powerful[-1]
    max_N = 2 * max_val
    
    # Create indicator array
    arr = np.zeros(max_N + 1, dtype=np.int32)
    for p in powerful:
        arr[p] = 1
    
    # FFT convolution
    from scipy.signal import fftconvolve
    counts = fftconvolve(arr, arr, mode='full')
    
    # Find top candidate N values
    # counts[N] = number of ordered pairs (i,j) with i+j=N
    # Unordered pairs with i<=j:
    # For N: unordered = (counts[N] + (1 if N even and N//2 in powerful_set else 0)) // 2
    
    # Find the N with maximum unordered count
    best_N = 2
    best_count = 0
    second_best_N = 2
    second_best_count = 0
    
    for N in range(2, min(max_N + 1, len(counts))):
        ordered = int(counts[N])
        if N % 2 == 0 and (N // 2) in powerful_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            second_best_N = best_N
            second_best_count = best_count
            best_N = N
            best_count = unordered
        elif unordered > second_best_count:
            second_best_N = N
            second_best_count = unordered
    
    print(f"Best N = {best_N}, count = {best_count}")
    print(f"Second best N = {second_best_N}, count = {second_best_count}")
    
    # Generate pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    pairs.sort()
    
    # Verify
    for a, b in pairs:
        assert a + b == best_N
        assert a <= b
        assert _is_powerful(a)
        assert _is_powerful(b)
    
    return {'N': best_N, 'pairs': pairs}
