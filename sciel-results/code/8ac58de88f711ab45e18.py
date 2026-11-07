import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    N = 10000000
    
    # Sieve of Eratosthenes for primes up to N
    # We need primes up to N for a, and up to N/2 for q (since b=2q <= N)
    # Actually, we just need to check primality of q and 3q+2.
    # Max q is ~3.33e6. Max p is ~1e7.
    
    # Efficient sieve
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    if time.time() > deadline - 5.0:
        return [11, 6]

    # We want to maximize q such that q is prime and 3q+2 is prime.
    # b = 2q <= N => q <= N//2
    # a = 3q+2 <= N => 3q <= N-2 => q <= (N-2)//3
    max_q = min(N // 2, (N - 2) // 3)
    
    best_q = 0
    # Iterate downwards from max_q to find the largest q quickly
    # Checking every number is slow in Python, but we only have ~3.3M checks.
    # We can optimize by only checking odd numbers > 2.
    
    # Check q=2 separately
    if is_prime[2] and is_prime[3*2+2]:
        best_q = 2
        
    for q in range(max_q - 1, 1, -2): # Only odd q
        if time.time() > deadline - 2.0:
            break
        if is_prime[q]:
            p = 3 * q + 2
            if is_prime[p]:
                best_q = q
                break # We want the largest q, so we stop at the first one found from top down

    if best_q > 0:
        a = 3 * best_q + 2
        b = 2 * best_q
        return [int(a), int(b)]
    else:
        return [11, 6]
