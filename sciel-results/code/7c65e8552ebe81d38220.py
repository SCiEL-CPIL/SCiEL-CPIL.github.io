"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _verify_subset_products_distinct(nums):
    """Check all 2^k subset products are distinct."""
    k = len(nums)
    products = set()
    # Use bitmasks: for each mask, compute product
    # More efficient: recursive generation
    prods = [1]
    for a in nums:
        new_prods = [p * a for p in prods]
        prods += new_prods
    if len(prods) != len(set(prods)):
        return False
    return True

def _verify_fast(nums):
    """Faster verification using incremental set check."""
    prods = {1}
    for a in nums:
        new_prods = set()
        for p in prods:
            val = p * a
            if val in prods or val in new_prods:
                return False
            new_prods.add(val)
        prods |= new_prods
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    primes = _primes_upto(P['n'])
    # Primes in 2..40: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes
    
    # Start with all primes (they always work)
    base = list(primes)
    best = base[:]
    
    # Try to add composites one by one
    composites = [a for a in range(2, P['n'] + 1) if a not in set(base)]
    
    # Greedy: try adding each composite, keep if valid
    current = base[:]
    for c in composites:
        if time.time() > deadline - 5:
            break
        candidate = current + [c]
        if _verify_fast(candidate):
            current = candidate
    
    best = current[:]
    
    # Now try to improve: use local search / swap
    # Try removing one element and adding a different one
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 10:
            break
        for i in range(len(best)):
            if time.time() > deadline - 10:
                break
            for c in composites:
                if c in best:
                    continue
                candidate = best[:i] + best[i+1:] + [c]
                candidate.sort()
                if _verify_fast(candidate):
                    best = candidate
                    improved = True
                    break
    
    # Try adding more elements if we have room
    for c in composites:
        if time.time() > deadline - 5:
            break
        if c in best:
            continue
        candidate = best + [c]
        if _verify_fast(candidate):
            best = candidate
    
    # Final local search: try all pairs of swaps
    if time.time() < deadline - 15:
        for i in range(len(best)):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, len(best)):
                if time.time() > deadline - 15:
                    break
                for c in composites:
                    if c in best:
                        continue
                    for d in composites:
                        if d in best or d == c:
                            continue
                        # Remove best[i] and best[j], add c and d
                        rest = best[:i] + best[i+1:j] + best[j+1:]
                        candidate = sorted(rest + [c, d])
                        if _verify_fast(candidate):
                            best = candidate
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    
    # Ensure return is a list of Python ints
    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
