"""Rewritten program to maximize coprime adjacent divisor pairs."""
import math
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
def _tau_perp(n):
    """Compute the number of adjacent coprime pairs in the sorted divisor list of n."""
    if n < 2:
        return 0
    # Generate all divisors
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort()
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j + 1]) == 1:
            count += 1
    return count

def _generate_divisors_from_primes(prime_list, exponents):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in zip(prime_list, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    divisors.sort()
    return divisors

def _tau_perp_fast(n, prime_list, exponents):
    """Compute tau_perp using precomputed factorization."""
    divisors = _generate_divisors_from_primes(prime_list, exponents)
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j + 1]) == 1:
            count += 1
    return count

def _factorize(n):
    """Return list of (prime, exponent) pairs for n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _search_best(deadline):
    """Search for the best n within the deadline."""
    best_n = 1
    best_score = 0
    
    # First, try the primorial approach with different numbers of primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    # Try squarefree numbers with various subsets of small primes
    for num_primes in range(2, 16):
        if time.time() > deadline:
            break
        subset = primes[:num_primes]
        n = 1
        for p in subset:
            n *= p
        if n > 10**18:
            break
        score = _tau_perp_fast(n, subset, [1] * num_primes)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Try numbers with some repeated prime factors
    # n = 2^a * 3^b * 5^c * ... 
    # Focus on small primes with varying exponents
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try a few structured candidates
    candidates = []
    
    # Primorial with one extra factor of 2
    for num_primes in range(2, 16):
        if time.time() > deadline:
            break
        subset = small_primes[:num_primes]
        n = 2 * 1
        for p in subset[1:]:
            n *= p
        if n <= 10**18:
            factors = _factorize(n)
            score = _tau_perp_fast(n, [f[0] for f in factors], [f[1] for f in factors])
            if score > best_score:
                best_score = score
                best_n = n
    
    # Try n = 2^a * (product of next k primes)
    for a in range(1, 60):
        if time.time() > deadline:
            break
        base = 2 ** a
        if base > 10**18:
            break
        # Multiply by as many next primes as fit
        n = base
        for p in small_primes[1:]:
            if n * p <= 10**18:
                n *= p
            else:
                break
        if n > 1:
            factors = _factorize(n)
            score = _tau_perp_fast(n, [f[0] for f in factors], [f[1] for f in factors])
            if score > best_score:
                best_score = score
                best_n = n
    
    # Try n = 2^a * 3^b * (product of next primes)
    for a in range(1, 40):
        if time.time() > deadline:
            break
        for b in range(1, 25):
            if time.time() > deadline:
                break
            base = (2 ** a) * (3 ** b)
            if base > 10**18:
                break
            n = base
            for p in small_primes[2:]:
                if n * p <= 10**18:
                    n *= p
                else:
                    break
            if n > 1:
                factors = _factorize(n)
                score = _tau_perp_fast(n, [f[0] for f in factors], [f[1] for f in factors])
                if score > best_score:
                    best_score = score
                    best_n = n
    
    return best_n, best_score

def construct():
    deadline = time.time() + 100  # Leave margin before 120s limit
    best_n, best_score = _search_best(deadline)
    return best_n
# EVOLVE-BLOCK-END
