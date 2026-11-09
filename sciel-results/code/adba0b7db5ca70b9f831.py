"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from collections import defaultdict
import bisect

MAX_N = 10_000_000
DEADLINE = 85.0
TARGET = Fraction(3, 2)

def construct():
    start_time = time.time()
    
    # Build SPF
    spf = list(range(MAX_N + 1))
    if MAX_N >= 1:
        spf[0] = 0
        spf[1] = 1
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() - start_time > DEADLINE - 30:
        return [11, 6]
    
    # Compute sigma using multiplicative property
    # sigma(1) = 1
    # For n > 1, let p = spf[n], m = n // p
    # If m % p == 0, then n = p^k * m' where p doesn't divide m'
    # We need to track the power of p in n
    # Alternative: use a modified sieve
    # sigma[n] can be computed as:
    # Let p = spf[n], m = n // p
    # If m == 1: sigma[n] = 1 + p
    # Else if spf[m] == p: 
    #   Let p^k || n, i.e., n = p^k * m' where p ∤ m'
    #   sigma[n] = sigma[m'] * (1 + p + ... + p^k)
    #   But we need to know k and m'
    # This requires tracking, which is complex
    # 
    # Simpler: use the additive sieve but with NumPy vectorization in chunks
    import numpy as np
    
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Process in batches to reduce Python overhead
    # But the fundamental issue is 10M iterations
    # Let's just do it and hope it's fast enough with NumPy
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        if d % 1000000 == 0:
            if time.time() - start_time > DEADLINE - 10:
                break
    
    if time.time() - start_time > DEADLINE - 5:
        return [11, 6]
    
    # Build dictionary
    sigma_dict = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        if s > 0:
            sigma_dict[s].append(n)
    
    if time.time() - start_time > DEADLINE - 3:
        return [11, 6]
    
    best_err = None
    best_pair = (11, 6)
    
    for s_val, nums in sigma_dict.items():
        if len(nums) < 2:
            continue
        if time.time() - start_time > DEADLINE - 1:
            break
        
        nums_sorted = sorted(nums)
        n_len = len(nums_sorted)
        
        for i in range(n_len):
            b = nums_sorted[i]
            target_a = 3 * b / 2
            pos = bisect.bisect_left(nums_sorted, target_a)
            candidates = []
            if pos < n_len:
                candidates.append(nums_sorted[pos])
            if pos > 0:
                candidates.append(nums_sorted[pos - 1])
            for a in candidates:
                if a == b:
                    continue
                err = abs(Fraction(a, b) - TARGET)
                if best_err is None or err < best_err:
                    best_err = err
                    best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
