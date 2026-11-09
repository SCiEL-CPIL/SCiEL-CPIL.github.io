import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80  # Safety margin before 90s limit
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Modified linear sieve to compute the powerful part for all integers up to MAX_N+1
    # The powerful part of n is the product of p^e for primes p with exponent e >= 2 in n's factorization
    
    N = MAX_N + 1  # We need up to MAX_N (since we check n and n+1 for n up to MAX_N-1)
    
    # Linear sieve arrays
    is_prime = [True] * (N + 1)
    primes = []
    spf = [0] * (N + 1)  # smallest prime factor
    spf[0] = 0
    spf[1] = 1
    
    # For each number, track:
    # - highest prime power: the largest prime power p^e (e>=1) dividing n
    # - powerful part: product of all p^e where e >= 2
    # We'll use a different approach: for each number, compute its powerful part directly
    
    # Actually, let's use the standard linear sieve to compute spf, then compute powerful parts
    
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            spf[i * p] = p
            if i % p == 0:
                break
    
    # Now compute powerful part for each number using spf
    pow_part = [1] * (N + 1)
    
    for n in range(2, N + 1):
        x = n
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        pow_part[n] = pp
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return 10000
    
    # Find n in [MIN_N, MAX_N-1] that maximizes pow_part[n] * pow_part[n+1]
    best_product = -1
    best_n_final = MIN_N
    
    for n in range(MIN_N, MAX_N):
        product = pow_part[n] * pow_part[n + 1]
        if product > best_product:
            best_product = product
            best_n_final = n
    
    return best_n_final
