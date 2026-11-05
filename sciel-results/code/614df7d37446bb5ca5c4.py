"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # Generate primes up to a reasonable limit
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0:2] = b'\x00\x00'
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(limit + 1) if sieve[i]]
    
    primes = get_primes(200)
    
    # Precompute score for a given n
    def calc_score(n):
        if n < 2:
            return 0
        # Get all divisors
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        divs.sort()
        score = 0
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score
    
    # Strategy: Try various combinations of primes with small exponents
    # Best known: squarefree products of first k primes, or 2^2 * product of next primes
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    best_score = calc_score(best_n)
    
    # Try: 2 * (product of first k-1 odd primes) for various k
    # This gives 2 * 3 * 5 * ... * p_k
    # Also try: 4 * (product of first k-2 odd primes) = 2^2 * 3 * 5 * ... * p_{k-1}
    
    # Generate candidate numbers
    candidates = []
    
    # Squarefree: product of first k primes
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2^2 * product of first (k-1) odd primes (3, 5, 7, ...)
    prod = 4
    for i, p in enumerate(primes[1:]):  # start from 3
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3^2 * product of first (k-2) primes from 5 onward
    prod = 2 * 9
    for i, p in enumerate(primes[2:]):  # start from 5
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5^2 * product of first (k-3) primes from 7 onward
    prod = 2 * 3 * 25
    for i, p in enumerate(primes[3:]):  # start from 7
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7^2 * product from 11 onward
    prod = 2 * 3 * 5 * 49
    for i, p in enumerate(primes[4:]):  # start from 11
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11^2 * product from 13 onward
    prod = 2 * 3 * 5 * 7 * 121
    for i, p in enumerate(primes[5:]):  # start from 13
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # Also try: 2 * 3 * 5 * 7 * 11 * 13^2 * product from 17
    prod = 2 * 3 * 5 * 7 * 11 * 169
    for i, p in enumerate(primes[6:]):  # start from 17
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17^2 * product from 19
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 289
    for i, p in enumerate(primes[7:]):  # start from 19
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19^2 * product from 23
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 361
    for i, p in enumerate(primes[8:]):  # start from 23
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23^2 * product from 29
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 529
    for i, p in enumerate(primes[9:]):  # start from 29
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29^2 * product from 31
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 841
    for i, p in enumerate(primes[10:]):  # start from 31
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31^2 * product from 37
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 961
    for i, p in enumerate(primes[11:]):  # start from 37
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37^2 * product from 41
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 1369
    for i, p in enumerate(primes[12:]):  # start from 41
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41^2 * product from 43
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 1681
    for i, p in enumerate(primes[13:]):  # start from 43
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43^2 * product from 47
    prod = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 1849
    for i, p in enumerate(primes[14:]):  # start from 47
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # Also try some with 2^3
    # 8 * 3 * 5 * 7 * ... 
    prod = 8
    for i, p in enumerate(primes[1:]):
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2^2 * 3^2 * 5 * 7 * ...
    prod = 4 * 9
    for i, p in enumerate(primes[2:]):
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2^2 * 3 * 5^2 * 7 * ...
    prod = 4 * 3 * 25
    for i, p in enumerate(primes[3:]):
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # Now evaluate all candidates
    for n in candidates:
        if n < 2 or n > 10**18:
            continue
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
