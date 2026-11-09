"""
Strategy: Exhaustive structured enumeration over n = 2^a * m where m is squarefree
with primes from a fixed set, using an efficient score computation.
We precompute scores for all valid (a, subset) combinations and pick the best.
"""
# EVOLVE-BLOCK-START
import time
import math
import itertools
from functools import lru_cache

DEADLINE = 145.0

# Primes to use for the squarefree part (excluding 2)
_ODD_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _score_from_divisors(ds):
    """Count coprime adjacent pairs in sorted divisor list."""
    if len(ds) < 2:
        return 0
    ds.sort()
    c = 0
    for i in range(1, len(ds)):
        if math.gcd(ds[i - 1], ds[i]) == 1:
            c += 1
    return c

def _score(n):
    """Compute tau_perp(n) by generating divisors."""
    if n < 2:
        return 0
    # Factor n
    factors = {}
    temp = n
    for p in _ODD_PRIMES:
        if p * p > temp:
            break
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        while temp % 2 == 0:
            factors[2] = factors.get(2, 0) + 1
            temp //= 2
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    # Generate divisors
    ds = [1]
    for p, a in factors.items():
        new_ds = []
        for d in ds:
            for k in range(a + 1):
                new_ds.append(d * (p ** k))
        ds = new_ds
    
    return _score_from_divisors(ds)

def construct():
    start = time.time()
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_s = _score(best_n)
    
    # Strategy: enumerate n = 2^a * m where m is squarefree product of odd primes
    # from a chosen subset, and a ranges from 0 to max_a.
    # We'll do this systematically.
    
    # Precompute powers of 2
    pow2 = [1]
    for a in range(1, 61):
        pow2.append(pow2[-1] * 2)
    
    # For each subset size k of odd primes, and for each a, compute the candidate
    # We want to be exhaustive over reasonable ranges.
    
    # Key insight: the best n likely has a moderate power of 2 and several small odd primes.
    # Let's enumerate:
    # - a from 0 to 55
    # - subsets of odd primes of size 2 to 10 from the first 24 odd primes
    
    # But that's too many subsets. Instead, use a greedy/structured approach:
    # For each a, find the best squarefree m (product of distinct odd primes) such that 2^a * m <= 10^18.
    # The best m for a given a will use the smallest odd primes.
    
    # Let's be smarter: enumerate all combinations of the first K odd primes (K up to ~15)
    # and for each combination, find the max a such that 2^a * product <= 10^18.
    
    # First, precompute the product of the first k odd primes for k=0..20
    prod_prefix = [1]
    for i in range(len(_ODD_PRIMES)):
        prod_prefix.append(prod_prefix[-1] * _ODD_PRIMES[i])
    
    # For each subset of the first K odd primes (K up to 15), and each a:
    # This is still 2^15 = 32768 subsets, times 60 values of a = ~2M candidates.
    # Each score computation is expensive. Let's optimize.
    
    # Alternative: for each a (0 to 55), the maximum product of odd primes we can use is 10^18 / 2^a.
    # The optimal m for a given a will be a product of the smallest odd primes.
    # So for each a, try m = product of first k odd primes for k = 0, 1, 2, ... until product > 10^18/2^a.
    
    # This gives at most 60 * 20 = 1200 candidates. Very fast.
    
    # But we might miss optimal subsets that aren't prefix-based.
    # Let's do both: prefix-based enumeration AND some structured variations.
    
    # Phase 1: Prefix-based enumeration
    for a in range(0, 60):
        if time.time() - start > DEADLINE - 5:
            break
        base = 1 << a
        if base > 10**18:
            break
        limit = 10**18 // base
        # Try m = product of first k odd primes
        m = 1
        for k in range(0, len(_ODD_PRIMES)):
            if k > 0:
                m *= _ODD_PRIMES[k - 1]
            if m > limit:
                break
            n = base * m
            if n < 2:
                continue
            if n > 10**18:
                break
            s = _score(n)
            if s > best_s:
                best_s = s
                best_n = n
    
    # Phase 2: More structured search - for each a, try different subsets
    # that swap out some larger primes for smaller ones.
    # Actually, let's just do a more thorough search over subsets of the first 12 odd primes.
    
    # Use itertools.combinations for subsets of size 2 to 10 from first 12 odd primes
    first_12_odd = _ODD_PRIMES[:12]
    
    for a in range(0, 55):
        if time.time() - start > DEADLINE - 5:
            break
        base = 1 << a
        if base > 10**18:
            break
        limit = 10**18 // base
        
        # Try subsets of first_12_odd of various sizes
        for k in range(2, 11):
            if time.time() - start > DEADLINE - 5:
                break
            for subset in itertools.combinations(first_12_odd, k):
                m = 1
                for p in subset:
                    m *= p
                if m > limit:
                    continue
                n = base * m
                if n < 2 or n > 10**18:
                    continue
                s = _score(n)
                if s > best_s:
                    best_s = s
                    best_n = n
    
    # Phase 3: Also try n with higher powers of odd primes (not just squarefree)
    # n = 2^a * 3^b * 5^c * ... for small exponents
    # This is a grid search over exponents.
    
    if time.time() - start < DEADLINE - 10:
        # Grid search over exponents of 2, 3, 5, 7
        # 2^a * 3^b * 5^c * 7^d <= 10^18
        max_a = 59
        max_b = int(math.log(10**18, 3))
        max_c = int(math.log(10**18, 5))
        max_d = int(math.log(10**18, 7))
        
        for a in range(0, max_a + 1):
            if time.time() - start > DEADLINE - 5:
                break
            p2 = 1 << a
            if p2 > 10**18:
                break
            for b in range(0, max_b + 1):
                if time.time() - start > DEADLINE - 5:
                    break
                p3 = 3 ** b
                if p2 * p3 > 10**18:
                    break
                for c in range(0, max_c + 1):
                    if time.time() - start > DEADLINE - 5:
                        break
                    p5 = 5 ** c
                    if p2 * p3 * p5 > 10**18:
                        break
                    for d in range(0, max_d + 1):
                        if time.time() - start > DEADLINE - 5:
                            break
                        p7 = 7 ** d
                        n = p2 * p3 * p5 * p7
                        if n > 10**18:
                            break
                        if n < 2:
                            continue
                        s = _score(n)
                        if s > best_s:
                            best_s = s
                            best_n = n
    
    # Phase 4: Add more primes to the grid search if time permits
    if time.time() - start < DEADLINE - 10:
        # n = 2^a * 3^b * 5^c * 7^d * 11^e
        max_e = int(math.log(10**18, 11))
        for a in range(0, 50):
            if time.time() - start > DEADLINE - 5:
                break
            p2 = 1 << a
            if p2 > 10**18:
                break
            for b in range(0, 38):
                if time.time() - start > DEADLINE - 5:
                    break
                p3 = 3 ** b
                if p2 * p3 > 10**18:
                    break
                for c in range(0, 26):
                    if time.time() - start > DEADLINE - 5:
                        break
                    p5 = 5 ** c
                    if p2 * p3 * p5 > 10**18:
                        break
                    for d in range(0, 22):
                        if time.time() - start > DEADLINE - 5:
                            break
                        p7 = 7 ** d
                        if p2 * p3 * p5 * p7 > 10**18:
                            break
                        for e in range(0, max_e + 1):
                            if time.time() - start > DEADLINE - 5:
                                break
                            p11 = 11 ** e
                            n = p2 * p3 * p5 * p7 * p11
                            if n > 10**18:
                                break
                            if n < 2:
                                continue
                            s = _score(n)
                            if s > best_s:
                                best_s = s
                                best_n = n
    
    # Phase 5: Randomized search with the fast score function
    import random
    random.seed(42)
    
    all_primes = [2] + _ODD_PRIMES[:20]
    
    while time.time() - start < DEADLINE - 2:
        # Random exponent vector
        n = 1
        valid = True
        for p in all_primes:
            if n > 10**18 // p:
                break
            # Random exponent: mostly 0 or 1, sometimes higher
            r = random.random()
            if r < 0.3:
                a = 0
            elif r < 0.7:
                a = 1
            elif r < 0.9:
                a = random.randint(2, 4)
            else:
                a = random.randint(2, 10)
            if a == 0:
                continue
            if n > 10**18 // (p ** a):
                valid = False
                break
            n *= p ** a
        
        if not valid or n < 2:
            continue
        
        s = _score(n)
        if s > best_s:
            best_s = s
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
