import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 80.0


def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct."""
    for p in prods:
        if (p * x) in prods:
            return False
    return True


def _try_add(prods, x):
    """Try to add x; returns (success, new_prods)."""
    if not _can_add(prods, x):
        return False, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, new_prods


def _build_prods(selected):
    """Build the full set of subset products for a list of integers."""
    prods = {1}
    for x in selected:
        if not _can_add(prods, x):
            return None
        new_prods = set(prods)
        for p in prods:
            new_prods.add(p * x)
        prods = new_prods
    return prods


def construct():
    if time.time() > _DEADLINE:
        return [2]

    # All primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Non-square-free composites in 2..40
    non_sqfree_composites = []
    for c in range(4, 41):
        if c in primes:
            continue
        is_sqfree = True
        d = 2
        n = c
        while d * d <= n:
            if n % (d * d) == 0:
                is_sqfree = False
                break
            d += 1
        if not is_sqfree:
            non_sqfree_composites.append(c)
    # [4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28, 32, 36, 40]

    # Build prods for just the primes (this is fast: 2^12 = 4096 elements)
    prime_prods = {1}
    for x in primes:
        _, prime_prods = _try_add(prime_prods, x)

    # Determine which non-sqfree composites can be added to the prime set
    # A composite x can be added iff for all p in prime_prods, p*x not in prime_prods
    # Since prime_prods contains only square-free numbers, and x has a squared
    # prime factor, p*x will have a squared prime factor, so p*x cannot be in
    # prime_prods. Thus ALL non-sqfree composites can be added to the prime set.
    # But we can only add up to 6 more (to reach 18 total).
    
    # However, after adding some composites, the prods set grows and may
    # create conflicts with other composites. We need to find a subset of 6
    # composites that are mutually compatible.
    
    # Strategy: try all C(14,6) = 3003 combinations, verify each.
    # But first, let's be smart: pre-filter which pairs of composites conflict.
    
    # Two composites x and y conflict (when added to primes) if there exist
    # a, b in prime_prods such that a*x = b*y or a*y = b*x.
    # Equivalently, x/y or y/x is a ratio of two elements in prime_prods.
    # Since prime_prods contains all square-free products of the 12 primes,
    # x/y is such a ratio iff x/y is square-free (a product of distinct primes
    # from our set, possibly with negative exponents meaning division).
    
    # Actually, simpler: just do the full check for each combination.
    # 3003 combinations * (12+6) additions * up to 2^18 = 262144 prods checks
    # That's too slow for naive approach. Let's optimize.
    
    # Better: incrementally build. For each combo, start with prime_prods
    # and add the 6 composites one by one, checking at each step.
    
    best_count = 12
    best_result = list(primes)
    
    # Try all C(14,6) combinations
    for combo in combinations(non_sqfree_composites, 6):
        if time.time() > _DEADLINE:
            break
        selected = list(primes) + list(combo)
        prods = prime_prods.copy()
        valid = True
        for x in combo:
            if not _can_add(prods, x):
                valid = False
                break
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
        if valid:
            return selected
    
    # If we didn't find 6, try 5
    for combo in combinations(non_sqfree_composites, 5):
        if time.time() > _DEADLINE:
            break
        selected = list(primes) + list(combo)
        prods = prime_prods.copy()
        valid = True
        for x in combo:
            if not _can_add(prods, x):
                valid = False
                break
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
        if valid:
            if len(selected) > best_count:
                best_count = len(selected)
                best_result = selected
    
    # Try 4
    for combo in combinations(non_sqfree_composites, 4):
        if time.time() > _DEADLINE:
            break
        selected = list(primes) + list(combo)
        prods = prime_prods.copy()
        valid = True
        for x in combo:
            if not _can_add(prods, x):
                valid = False
                break
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
        if valid:
            if len(selected) > best_count:
                best_count = len(selected)
                best_result = selected
    
    return best_result if best_count >= 1 else [2]
