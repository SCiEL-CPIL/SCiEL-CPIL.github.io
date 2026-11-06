import math
import time
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(100)

def _tau_perp_from_factorization(primes, exponents):
    """Compute tau_perp score from prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        current = list(divisors)
        for k in range(1, e + 1):
            for d in current:
                divisors.append(d * (p ** k))
    divisors.sort()
    if len(divisors) < 2:
        return 0
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    max_n = 10**18
    primes = _PRIMES[:25]
    
    # Try primorial structures with various prime counts
    # and some exponent variations
    
    # First: pure primorials (all exponents 1)
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > max_n:
            break
        prod *= p
        exps = [1] * (i + 1)
        act_primes = primes[:i+1]
        score = _tau_perp_from_factorization(act_primes, exps)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Now try variations: increase some exponents
    # Strategy: for each primorial, try bumping one prime's exponent to 2
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > max_n:
            break
        prod *= p
    
    # Try adding a squared prime to smaller primorials
    base_prod = 1
    for i, p in enumerate(primes):
        if base_prod * p > max_n:
            break
        base_prod *= p
        # Try: use this primorial but square one of the smaller primes
        for j in range(i + 1):
            if time.time() > deadline:
                return best_n
            # Exponents: all 1 except prime j has exponent 2
            exps = [1] * (i + 1)
            exps[j] = 2
            candidate = base_prod * primes[j]  # multiply by extra p_j
            if candidate > max_n:
                continue
            act_primes = primes[:i+1]
            score = _tau_perp_from_factorization(act_primes, exps)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    # Try: primorial up to some prime, then add another prime squared
    # e.g., 2*3*5*7*11*13*17*19*23*29 * 31^2
    base_prod = 1
    for i, p in enumerate(primes):
        if base_prod * p > max_n:
            break
        base_prod *= p
        if time.time() > deadline:
            return best_n
        # Try adding next prime squared
        if i + 1 < len(primes):
            next_p = primes[i + 1]
            candidate = base_prod * next_p * next_p
            if candidate <= max_n:
                exps = [1] * (i + 1) + [2]
                act_primes = primes[:i+2]
                score = _tau_perp_from_factorization(act_primes, exps)
                if score > best_score:
                    best_score = score
                    best_n = candidate
    
    return best_n
