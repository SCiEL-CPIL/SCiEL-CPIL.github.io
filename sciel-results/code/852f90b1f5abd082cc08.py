import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 123457
    
    # Compute LPF (largest prime factor) for all numbers up to N
    # Efficient sieve: for each prime p, mark all multiples with p
    lpf = [1] * (N + 1)
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            # i is prime, mark all multiples
            for j in range(i, N + 1, i):
                is_prime[j] = False
                lpf[j] = i  # i is the largest prime factor seen so far for j
    
    # Now do the linear scan to find the optimal pair
    # Since N is odd, one of a,b is even and the other is odd
    best_val = max(lpf[1], lpf[N - 1])  # a=1, b=N-1
    best_a = 1
    best_b = N - 1
    
    for a in range(2, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            # If we've found a very good value, we can potentially stop early
            # but let's continue to be safe
    
    return [best_a, best_b]
