import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need to compute up to n+1 where n can be MAX_N
    
    # Sieve-based computation of powerful_part for all numbers up to LIMIT
    # powerful_part[n] = product of p^e for all prime p with exponent e>=2 in factorization of n
    # Initialize to 1 for all
    powerful_part = [1] * (LIMIT + 1)
    
    # For each prime p, for each power p^e with e>=2, multiply into powerful_part for all multiples of p^e
    # But we need to be careful: if p^e divides n, we include the full p^e (not just one factor)
    # So for n, powerful_part(n) = product over primes p | n of (p^e if e>=2 else 1)
    
    # Better approach: for each number n, factorize it and compute powerful part.
    # But that's O(N * sqrt(N)) which is too slow.
    
    # Alternative: use a modified sieve.
    # For each prime p, find all powers p^2, p^3, ... <= LIMIT
    # For each such power p^e, for each multiple m of p^e, we need to determine if p^(e+1) also divides m
    # If not, then the exponent of p in m is exactly e, so we multiply powerful_part[m] by p^e
    # If yes, we don't multiply yet (wait for higher power)
    
    # This is tricky. Let me think differently.
    
    # For each n, the powerful part is determined by the factorization.
    # We can use a smallest-prime-factor sieve to factorize each number quickly.
    
    # Build smallest prime factor (spf) array
    spf = list(range(LIMIT + 1))  # spf[i] = smallest prime factor of i
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute powerful_part for each n using spf
    for n in range(2, LIMIT + 1):
        temp = n
        pp = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        powerful_part[n] = pp
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return MIN_N
    
    # Scan all n in [MIN_N, MAX_N] and find max powerful_part[n] * powerful_part[n+1]
    best_product = 0
    best_n = MIN_N
    
    for n in range(MIN_N, MAX_N + 1):
        product = powerful_part[n] * powerful_part[n + 1]
        if product > best_product:
            best_product = product
            best_n = n
    
    # Safety check
    if best_n < MIN_N or best_n > MAX_N:
        best_n = MIN_N
    
    return best_n
