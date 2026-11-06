import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin

    # Sieve primes up to 10^6
    SIEVE_LIMIT = 10_000_000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = 0
    
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    
    best_k = 1
    best_p = 2

    # We will iterate through candidate primes p.
    # To maximize efficiency, we should probably try smaller primes first?
    # Or larger?
    # The probability of a number being p-smooth is rho(2) ~ 0.3.
    # This is independent of p (as long as p is large enough).
    # However, the cost of sieving is proportional to pi(p).
    # So smaller p is cheaper to check.
    # Let's check primes in increasing order.
    
    # Optimization: We can skip p if p^2+1 is not p-smooth.
    # The sieve will tell us this.
    
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        p2 = p * p
        L = p2 + 1
        R = p2 + 200
        
        # Initialize remainders
        # rem[i] corresponds to L+i
        rem = [L + i for i in range(200)]
        
        # Sieve with primes q <= p
        # We need to iterate through primes list and stop when q > p
        # Since primes is sorted, we can find the index
        # But p is from primes list, so we can just iterate until q > p
        
        for q in primes:
            if q > p:
                break
                
            # Find first multiple of q in [L, R]
            # first = L + (q - L % q) % q
            # If L % q == 0, first = L. Else first = L + (q - L%q)
            
            # Optimization: If q > R, no multiples. But q <= p < sqrt(R) roughly?
            # p^2 = L-1. q <= p. q*q <= p^2 < L.
            # So multiples exist.
            
            # Calculate first multiple
            # Use integer arithmetic
            first = L + (q - L % q) % q
            
            # Iterate through multiples
            # step = q
            # for m in range(first, R + 1, q):
            #     idx = m - L
            #     while rem[idx] % q == 0:
            #         rem[idx] //= q
            
            # This inner while loop is the bottleneck.
            # We can optimize by checking if rem[idx] is 1 first.
            
            for m in range(first, R + 1, q):
                idx = m - L
                if rem[idx] == 1:
                    continue
                while rem[idx] % q == 0:
                    rem[idx] //= q
                    
        # Now check smoothness
        # A number is p-smooth if rem[i] == 1
        # Because we divided out all prime factors <= p.
        # If rem[i] > 1, it has a prime factor > p.
        
        k = 0
        for i in range(200):
            if rem[i] == 1:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
