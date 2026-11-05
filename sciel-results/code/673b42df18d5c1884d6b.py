import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s margin
    
    # The problem asks to minimize the largest prime factor of n(n+1)
    # for n in [10000, 1000000].
    # This is equivalent to minimizing max(lpf(n), lpf(n+1)).
    
    # We need lpf for numbers up to 1000001 (since n can be 1000000, n+1 is 1000001).
    upper = 1000001
    
    # Sieve for Largest Prime Factor
    # lpf[i] will store the largest prime factor of i.
    # Initialize to 0.
    lpf = np.zeros(upper + 1, dtype=np.int32)
    
    # Iterate through potential prime factors
    # If lpf[i] is 0, i is prime (not divisible by any smaller prime).
    for i in range(2, upper + 1):
        if time.time() > deadline:
            # Fallback: return a valid n, e.g., 10000
            # This should never happen for N=1e6
            return 10000
            
        if lpf[i] == 0:
            # i is prime. Set lpf for all multiples of i to i.
            # Since we iterate i in increasing order, the last assignment
            # to lpf[j] will be by the largest prime factor of j.
            lpf[i::i] = i
            
    if time.time() > deadline:
        return 10000

    # Find n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # n ranges from 10000 to 1000000 inclusive.
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 100001 to 1000001
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Compute the maximum of the two LPFs for each pair
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    
    # Map index back to n
    best_n = 10000 + min_idx
    
    return int(best_n)
