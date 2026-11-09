"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    MAX_N = 1000001  # we need n up to 1000000, so n+1 up to 1000001
    
    # Precompute the largest perfect square dividing each number
    # square_part[i] = largest perfect square that divides i
    square_part = [1] * (MAX_N + 1)
    
    # For each perfect square s = k^2, mark all multiples of s
    # The square part of a number n is the product of p^(2*floor(e/2)) for each prime p with exponent e
    # Equivalent: the largest perfect square dividing n
    
    # Use sieve approach: for each k from 2 to sqrt(MAX_N), for each multiple of k^2,
    # we can accumulate the square factor.
    
    # Better approach: factor each number using a linear sieve to get exponents,
    # then compute the square part.
    
    # Linear sieve to get smallest prime factor and exponent
    spf = [0] * (MAX_N + 1)  # smallest prime factor
    primes = []
    spf[0] = 0
    spf[1] = 1
    
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p
    
    # Now compute square_part for each number
    # square_part[n] = product of p^(2*floor(e/2)) for each prime factor p with exponent e
    square_part = [1] * (MAX_N + 1)
    square_part[0] = 1
    square_part[1] = 1
    
    for i in range(2, MAX_N + 1):
        n = i
        result = 1
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                # p^(2*floor(count/2))
                exp = 2 * (count // 2)
                result *= (p ** exp)
        square_part[i] = result
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return 10000
    
    # Now scan for max product of adjacent square parts
    # We want n in [10000, 1000000], so n+1 in [10001, 1000001]
    best_n = 10000
    best_product = square_part[10000] * square_part[10001]
    
    for n in range(10000, 1000001):
        prod = square_part[n] * square_part[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline periodically
        if n % 50000 == 0:
            if time.time() - start_time > DEADLINE:
                break
    
    return best_n
# EVOLVE-BLOCK-END
