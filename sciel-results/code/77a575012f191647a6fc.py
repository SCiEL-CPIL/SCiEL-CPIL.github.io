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

def _max_independent_exponents(max_exp):
    """Find largest subset of {1..max_exp} with distinct subset sums."""
    best_size = 0
    best_set = []
    # Try all subsets? max_exp is small (<=5 for p=2, <=3 for p=3)
    from itertools import combinations
    for r in range(max_exp, 0, -1):
        found = False
        for comb in combinations(range(1, max_exp + 1), r):
            sums = set()
            ok = True
            # Check distinct subset sums
            # Number of subsets is 2^r. r is small.
            for mask in range(1 << r):
                s = 0
                for i in range(r):
                    if mask & (1 << i):
                        s += comb[i]
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if ok:
                best_set = list(comb)
                best_size = r
                found = True
                break
        if found:
            break
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    primes = _primes_upto(P['n'])
    
    # Strategy: Use prime powers to maximize independence
    # For each prime, pick a set of exponents with distinct subset sums
    selected = []
    
    for p in primes:
        # Find max power p^k <= 40
        max_k = 1
        val = p
        while val * p <= P['n']:
            val *= p
            max_k += 1
            
        # Get best exponents
        exps = _max_independent_exponents(max_k)
        
        for k in exps:
            selected.append(p ** k)
            
    best = sorted(selected)
    
    # Quick local search to see if we can add any mixed number
    # Only try if we have time
    if time.time() < deadline - 5:
        all_nums = list(range(2, P['n'] + 1))
        current_set = set(best)
        
        # Try adding one element
        for c in all_nums:
            if time.time() > deadline - 5:
                break
            if c in current_set:
                continue
            candidate = best + [c]
            if _verify_fast(candidate):
                best = candidate
                current_set.add(c)
                
    # Ensure return is a list of Python ints
    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
