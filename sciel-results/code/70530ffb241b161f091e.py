import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Precompute largest prime factor for all numbers up to target
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # Since we iterate i from 2 to target in increasing order,
    # and set lpf[j] = i for multiples j of i,
    # the final value stored is the largest prime factor.
    for i in range(2, target + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
    
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Search for the pair (a, b) with a+b=target that minimizes max(P(a), P(b))
    # Since a and b are interchangeable in the objective, we only need to check
    # a from 1 to target//2 (inclusive).
    half = target // 2
    for a in range(1, half + 1):
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            # Theoretical minimum for max(P(a), P(b)) when a+b is odd and a,b > 1:
            # One is even (P>=2), one is odd (P>=3). So min possible is 3.
            if best_max_p == 3:
                break
        
        # Time guard (unlikely to trigger for N=123457, but required by directive)
        if time.time() > deadline:
            break
    
    return [best_a, best_b]
