import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 1000001  # We need lpf for n+1 where n can be 1000000
    
    # Sieve to find primes and also build largest prime factor array
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    
    # Initialize lpf for primes to themselves, composites will be updated
    # We can do this in a single pass using a modified sieve
    
    # Standard approach:
    # 1. Identify primes
    # 2. For each prime p, update lpf for all multiples of p
    
    # Let's use a simple sieve to get primes first, or combine.
    # Combined approach:
    # is_prime = [True] * (N + 1)
    # is_prime[0] = is_prime[1] = False
    # for i in range(2, int(math.isqrt(N)) + 1):
    #     if is_prime[i]:
    #         for j in range(i*i, N+1, i):
    #             is_prime[j] = False
    #             # We can't just set lpf[j] = i here because i might not be the largest.
    #             # But if we iterate i in increasing order, and for each i that is prime,
    #             # we set lpf[j] = i for all multiples j, then at the end lpf[j] is the largest prime factor.
    
    # Let's do it cleanly:
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    
    # Now populate lpf
    # Iterate through all numbers, if it's a prime, mark its multiples
    # Actually, we can just iterate p from 2 to N. If is_prime[p], then for j in p, 2p, ..., lpf[j] = p.
    # This is O(N log log N)
    
    for p in range(2, N + 1):
        if is_prime[p]:
            for j in range(p, N + 1, p):
                lpf[j] = p
    
    # Now lpf[i] contains the largest prime factor of i for all i >= 2.
    # For i=1, lpf[1] is 0, but n>=10000 so we don't care.
    
    best_n = 10000
    best_val = float('inf')
    
    # Scan for the optimal n
    for n in range(10000, 1000001):
        # Check deadline
        if time.time() > deadline:
            break
            
        # Largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        # Since n and n+1 are coprime, the prime factors of the product are the union of prime factors of n and n+1.
        # The largest is the max of the largest of each.
        current_max_lpf = max(lpf[n], lpf[n+1])
        
        if current_max_lpf < best_val:
            best_val = current_max_lpf
            best_n = n
            
    return best_n
