# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
from functools import lru_cache

def _prime_sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _factorize(n, primes):
    """Factor n using given primes. Returns dict of prime->exponent."""
    result = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            result[p] = result.get(p, 0) + 1
            n //= p
    if n > 1:
        result[n] = result.get(n, 0) + 1
    return result

def _sigma_from_factors(factors):
    """Compute sigma from factorization dict."""
    result = 1
    for p, e in factors.items():
        result *= (p**(e + 1) - 1) // (p - 1)
    return result

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s kill

    # Precompute k*sigma(k) for k up to K_MAX
    K_MAX = 5000
    products = {}  # product -> list of k values
    for k in range(1, K_MAX + 1):
        # Compute sigma(k) via trial division
        n = k
        sigma_val = 1
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                term = 1
                power = 1
                while temp % d == 0:
                    temp //= d
                    power *= d
                    term += power
                sigma_val *= term
            d += 1 if d == 2 else 2
        if temp > 1:
            sigma_val *= (1 + temp)
        prod = k * sigma_val
        if prod not in products:
            products[prod] = []
        products[prod].append(k)

    # Find best base M from precomputed products
    best_M = None
    best_k_list = []
    for M, klist in products.items():
        if M <= 10**18 and len(klist) > len(best_k_list):
            best_M = M
            best_k_list = list(klist)

    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Try to expand by finding k values beyond K_MAX
    # For the best M, check if there are additional k values
    # k*sigma(k) = M means sigma(k) = M/k, so k must divide M
    # Also k <= sqrt(M)
    M = best_M
    k_list = list(best_k_list)
    k_set = set(k_list)

    # Factor M to find all divisors k where k <= sqrt(M) and k*sigma(k) = M
    # First, find divisors of M
    sqrtM = math.isqrt(M)
    if M <= 10**18:
        # Get divisors
        divisors = []
        for d in range(1, sqrtM + 1):
            if M % d == 0:
                divisors.append(d)
                other = M // d
                if other != d and other <= sqrtM:
                    divisors.append(other)
        # For each divisor k, check if k*sigma(k) == M
        for k in divisors:
            if k in k_set:
                continue
            if k * k > M:
                continue
            # Compute sigma(k)
            temp = k
            sigma_val = 1
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    term = 1
                    power = 1
                    while temp % d == 0:
                        temp //= d
                        power *= d
                        term += power
                    sigma_val *= term
                d += 1 if d == 2 else 2
            if temp > 1:
                sigma_val *= (1 + temp)
            if k * sigma_val == M:
                k_list.append(k)
                k_set.add(k)

    # Now try to find a larger M that has more solutions
    # Strategy: take the best product P from small k values
    # and see if multiplying by some factor gives more solutions
    # Actually, let's try a different approach:
    # Look for M where multiple "types" of k work
    
    # Better approach: try M values that are products of many small prime powers
    # Let's try M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f ...
    # where a, b, c, ... are chosen so that many k work
    
    # Alternative: search for M by trying all M up to some limit
    # and counting solutions
    
    # Let's try a targeted search: for M up to 10^9, count k*sigma(k)=M
    # This is feasible with precomputation up to sqrt(10^9) ~ 31623
    
    # But that's too slow for full range. Let's stick with our best so far
    # and try to improve by checking if we can find more solutions for existing M

    # Try: for our best M, also check k values where k*sigma(k) might equal M
    # but we already did that above via divisors
    
    # Let's also try: find M that appears as k*sigma(k) for k in a larger range
    # by checking specific structured k values
    
    # Check if time allows for a broader search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Try to find better M by checking more k values
        # Focus on k up to 20000
        for k in range(K_MAX + 1, 20001):
            if time.time() - start_time > DEADLINE - 5:
                break
            temp = k
            sigma_val = 1
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    term = 1
                    power = 1
                    while temp % d == 0:
                        temp //= d
                        power *= d
                        term += power
                    sigma_val *= term
                d += 1 if d == 2 else 2
            if temp > 1:
                sigma_val *= (1 + temp)
            prod = k * sigma_val
            if prod <= 10**18 and prod in products:
                products[prod].append(k)
                if len(products[prod]) > len(k_list):
                    best_M = prod
                    best_k_list = list(products[prod])
                    k_list = list(products[prod])
                    k_set = set(k_list)

    # Verify all solutions
    verified = []
    for k in k_list:
        if k < 1 or k > math.isqrt(best_M):
            continue
        # Recompute sigma(k)
        temp = k
        sigma_val = 1
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                term = 1
                power = 1
                while temp % d == 0:
                    temp //= d
                    power *= d
                    term += power
                sigma_val *= term
            d += 1 if d == 2 else 2
        if temp > 1:
            sigma_val *= (1 + temp)
        if k * sigma_val == best_M:
            verified.append(k)
    
    if not verified:
        verified = [336, 372, 434]
        best_M = 333312

    # Ensure distinct and sorted
    verified = sorted(set(verified))
    
    # Ensure types are exactly int and list
    M_final = int(best_M)
    k_final = [int(k) for k in verified]
    
    return {'M': M_final, 'k': k_final}
# EVOLVE-BLOCK-END
