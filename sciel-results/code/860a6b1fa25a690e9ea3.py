"""Rewritten program for maximizing tau_perp(n)."""
import math
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
def _score(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors of n."""
    if n < 2:
        return 0
    # Factorize n
    factors = {}
    d = 2
    x = n
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    
    primes = sorted(factors.keys())
    exps = [factors[p] for p in primes]
    
    # Generate all divisors as tuples of exponents, then map to values
    divisors = [1]
    for i, p in enumerate(primes):
        e = exps[i]
        current = []
        for j in range(len(divisors)):
            val = divisors[j]
            for k in range(e + 1):
                current.append(val * (p ** k))
        divisors = current
    
    divisors.sort()
    
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            score += 1
    return score


def _try_candidates():
    """Try a set of carefully chosen candidate n values and return the best."""
    candidates = []
    
    # First 15 primes
    primes15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Exponent tuples to try (decreasing prime, increasing exponent)
    # Format: list of exponents for primes in order
    exp_tuples = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # product of first 15 primes
        [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [9, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [9, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [9, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 5, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 5, 3, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 5, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1],
    ]
    
    for exps in exp_tuples:
        if len(exps) != len(primes15):
            continue
        n = 1
        valid = True
        for i, p in enumerate(primes15):
            if exps[i] > 0:
                n *= p ** exps[i]
            if n > 10**18:
                valid = False
                break
        if valid and n >= 2:
            candidates.append(n)
    
    # Also try with fewer primes but higher exponents
    primes12 = primes15[:12]
    exp_tuples12 = [
        [14, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [13, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [13, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [12, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [12, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [12, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [11, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [11, 4, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 3, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 5, 4, 4, 3, 2, 2, 1, 1, 1, 1, 1],
        [10, 5, 4, 3, 4, 3, 2, 1, 1, 1, 1, 1],
        [10, 4, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1],
        [9, 6, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1],
        [9, 6, 5, 4, 5, 4, 3, 2, 1, 1, 1, 1],
        [9, 5, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 2, 1, 1, 1],
        [8, 7, 6, 5, 6, 5, 4, 3, 2, 1, 1, 1],
        [8, 6, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1],
        [7, 8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1],
        [7, 8, 7, 6, 7, 6, 5, 4, 3, 1, 1, 1],
        [7, 7, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1],
    ]
    
    for exps in exp_tuples12:
        if len(exps) != len(primes12):
            continue
        n = 1
        valid = True
        for i, p in enumerate(primes12):
            if exps[i] > 0:
                n *= p ** exps[i]
            if n > 10**18:
                valid = False
                break
        if valid and n >= 2:
            candidates.append(n)
    
    # Also try with 9 primes
    primes9 = primes15[:9]
    exp_tuples9 = [
        [18, 1, 1, 1, 1, 1, 1, 1, 1],
        [17, 2, 1, 1, 1, 1, 1, 1, 1],
        [16, 3, 2, 1, 1, 1, 1, 1, 1],
        [16, 2, 3, 2, 1, 1, 1, 1, 1],
        [15, 4, 3, 3, 2, 1, 1, 1, 1],
        [15, 4, 3, 2, 3, 2, 1, 1, 1],
        [14, 5, 4, 3, 3, 2, 1, 1, 1],
        [14, 5, 4, 3, 2, 3, 2, 1, 1],
        [13, 6, 5, 4, 3, 3, 2, 1, 1],
        [13, 6, 5, 4, 3, 2, 3, 2, 1],
        [12, 7, 6, 5, 4, 3, 3, 2, 1],
        [12, 7, 6, 5, 4, 3, 2, 3, 2],
        [12, 6, 7, 6, 5, 4, 3, 3, 2],
        [11, 8, 7, 6, 6, 4, 4, 3, 2],
        [11, 8, 7, 6, 5, 5, 4, 4, 3],
    ]
    
    for exps in exp_tuples9:
        if len(exps) != len(primes9):
            continue
        n = 1
        valid = True
        for i, p in enumerate(primes9):
            if exps[i] > 0:
                n *= p ** exps[i]
            if n > 10**18:
                valid = False
                break
        if valid and n >= 2:
            candidates.append(n)
    
    # Evaluate all candidates
    best_n = 2
    best_score = 0
    for n in candidates:
        if n < 2 or n > 10**18:
            continue
        s = _score(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n


def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try the pre-computed candidates
    best_n = _try_candidates()
    
    # If we have time, we could do a more extensive search
    # But the pre-computed candidates should be good enough
    return best_n
# EVOLVE-BLOCK-END
