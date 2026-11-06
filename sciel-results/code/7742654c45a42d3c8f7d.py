import time
import math

TARGET = 123457

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute smallest prime factor (SPF) up to TARGET
    # This allows fast determination of the largest prime factor.
    spf = list(range(TARGET + 1))
    for i in range(2, int(math.isqrt(TARGET)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, TARGET + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(n):
        if n == 1:
            return 1
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p
            while n % p == 0:
                n //= p
        return max_pf

    best_val = float('inf')
    best_pair = (1, TARGET - 1)
    
    # Iterate through all possible splits
    for a in range(1, TARGET):
        if time.time() > deadline:
            break
            
        b = TARGET - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        
        current_max = max(pa, pb)
        
        if current_max < best_val:
            best_val = current_max
            best_pair = (a, b)
            
            # Optimization: The minimum possible value for max(P(a), P(b)) is 3.
            # Reasoning: a+b is odd, so one is even, one is odd.
            # P(even) >= 2 (if even > 1). P(odd) >= 3 (if odd > 1).
            # If one is 1, P(1)=1. The other is 123456 (even), P(123456) is large.
            # If both > 1, max(P(a), P(b)) >= max(2, 3) = 3.
            # So 3 is the global minimum possible.
            if best_val == 3:
                break

    return best_pair
