"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Precomputed primes in 2..40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
PRIME_SET = set(PRIMES)

# Exponent vectors for numbers 2..40
# Map number to tuple of exponents for each prime in PRIMES
EXP_VECTORS = {}
for i in range(2, 41):
    vec = [0] * len(PRIMES)
    n = i
    for j, p in enumerate(PRIMES):
        while n % p == 0:
            vec[j] += 1
            n //= p
    if n > 1:
        # n is a prime > 37? No, max is 40.
        # But if n is prime not in PRIMES, it's a new dimension.
        # However, all primes <= 40 are in PRIMES.
        pass
    EXP_VECTORS[i] = tuple(vec)

def _subset_sums_distinct(vecs):
    """
    Check if all subset sums of the given exponent vectors are distinct.
    vecs: list of tuples (exponents)
    """
    sums = {tuple([0]*len(vecs[0]))}
    for v in vecs:
        new_sums = set()
        for s in sums:
            new_s = tuple(s[i] + v[i] for i in range(len(s)))
            if new_s in sums:
                return False
            new_sums.add(new_s)
        sums |= new_sums
    return True

def _is_valid(set_):
    """
    Verify all 2**k subset products are distinct.
    """
    if not set_:
        return True
    vecs = [EXP_VECTORS[x] for x in set_]
    return _subset_sums_distinct(vecs)

def _greedy_max_set(candidates, deadline, start_set=None):
    """
    Greedy: try to build largest valid set by adding elements one by one.
    """
    best = list(start_set) if start_set else []
    best_vecs = [EXP_VECTORS[x] for x in best]
    
    # Sort candidates to prioritize "independent" looking ones?
    # Primes are safest.
    # Then prime powers.
    # Then others.
    
    # Try to add elements
    for x in candidates:
        if time.time() > deadline:
            break
        if x in best:
            continue
        trial_vecs = best_vecs + [EXP_VECTORS[x]]
        if _subset_sums_distinct(trial_vecs):
            best.append(x)
            best_vecs.append(EXP_VECTORS[x])
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin
    
    # Start with a known good 16-element set
    initial_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify initial set
    if not _is_valid(initial_set):
        # Fallback to primes if initial is broken (should not happen)
        initial_set = list(PRIMES)
        
    best_set = list(initial_set)
    
    # Try to improve by adding elements
    candidates = [x for x in range(2, 41) if x not in best_set]
    
    # Order candidates: Primes first (though all are in set), then powers, then others
    # Actually, all primes are in the set.
    # So candidates are composites.
    
    # Try greedy addition
    improved_set = _greedy_max_set(candidates, deadline, start_set=best_set)
    if len(improved_set) > len(best_set):
        best_set = improved_set
        
    # Try local search: swap one element with another to allow adding more?
    # This is complex. Simple greedy might not find 17 if it exists.
    # But given the structure, 16 is likely the max.
    # We will just return the best found.
    
    # Ensure types are correct
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
