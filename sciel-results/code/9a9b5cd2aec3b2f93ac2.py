# EVOLVE-BLOCK-START
from math import prod, gcd, isqrt
import itertools
from functools import lru_cache

_LIMIT = 10**18

# Generate more primes
def _sieve(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

_SMALL_PRIMES = _sieve(1500)

def _divisors_sorted(primes, exps):
    """Generate all divisors of n = prod(p_i ** e_i) in sorted order."""
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(divs):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_from_factors(primes, exps):
    """Compute tau_perp from prime factorization."""
    divs = _divisors_sorted(primes, exps)
    return _tau_perp(divs)

def _search_square_free(k, primes_pool):
    """Search for best square-free n with exactly k primes from primes_pool."""
    best_score = 0
    best_primes = None
    
    # Start with the k smallest primes
    base_primes = primes_pool[:k]
    base_n = prod(base_primes)
    if base_n > _LIMIT:
        return 0, None
    
    s = _score_from_factors(base_primes, [1] * k)
    if s > best_score:
        best_score = s
        best_primes = list(base_primes)
    
    # Try replacing individual primes with larger ones
    # Focus on replacing larger-indexed primes first (less impact on product)
    for pos in range(k - 1, -1, -1):
        for q in primes_pool[k:k + 50]:
            if q <= base_primes[pos]:
                continue
            new_primes = list(base_primes)
            new_primes[pos] = q
            new_n = prod(new_primes)
            if new_n > _LIMIT:
                break
            s2 = _score_from_factors(new_primes, [1] * k)
            if s2 > best_score:
                best_score = s2
                best_primes = new_primes
                base_primes = new_primes
    
    return best_score, best_primes

def _search_mixed_exponents(k, primes_pool):
    """Search for best n with k distinct primes, some with exponent 2."""
    best_score = 0
    best_factors = None
    
    # Try putting exponent 2 on the smallest prime
    base_primes = primes_pool[:k]
    for exp_pos in range(min(k, 3)):  # Try exponent 2 on first few primes
        exps = [1] * k
        exps[exp_pos] = 2
        n = prod(p ** e for p, e in zip(base_primes, exps))
        if n > _LIMIT:
            continue
        s = _score_from_factors(base_primes, exps)
        if s > best_score:
            best_score = s
            best_factors = (list(base_primes), list(exps))
    
    # Also try with slightly different prime sets
    for k2 in range(k - 1, k + 2):
        if k2 < 2 or k2 > len(primes_pool):
            continue
        primes = primes_pool[:k2]
        for exp_pos in range(min(k2, 2)):
            exps = [1] * k2
            exps[exp_pos] = 2
            n = prod(p ** e for p, e in zip(primes, exps))
            if n > _LIMIT:
                continue
            s = _score_from_factors(primes, exps)
            if s > best_score:
                best_score = s
                best_factors = (list(primes), list(exps))
    
    return best_score, best_factors

def construct():
    best_n = 2
    best_score = 0
    
    # Search square-free with various k
    for k in range(16, 9, -1):
        if k > len(_SMALL_PRIMES):
            continue
        s, primes = _search_square_free(k, _SMALL_PRIMES)
        if s > best_score and primes:
            best_score = s
            best_n = prod(primes)
    
    # Search mixed exponents
    for k in range(15, 9, -1):
        if k > len(_SMALL_PRIMES):
            continue
        s, factors = _search_mixed_exponents(k, _SMALL_PRIMES)
        if s > best_score and factors:
            primes, exps = factors
            n = prod(p ** e for p, e in zip(primes, exps))
            if n <= _LIMIT:
                best_score = s
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
