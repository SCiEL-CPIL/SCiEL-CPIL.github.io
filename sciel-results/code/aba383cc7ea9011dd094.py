import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 85.0  # safety margin under 90s


def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct.
    Uses modular arithmetic: p*x in prods iff x divides q and q//x in prods for some q.
    Actually simpler: just check p*x for each p, but use the division trick for speed."""
    for p in prods:
        q = p * x
        if q in prods:
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
    """Build the full set of subset products for a list of integers.
    Returns None if any collision detected."""
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
    non_sqfree = []
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
            non_sqfree.append(c)
    # non_sqfree = [4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28, 32, 36, 40]

    # Build prods for just the primes (2^12 = 4096 elements, fast)
    prime_prods = {1}
    for x in primes:
        _, prime_prods = _try_add(prime_prods, x)

    # All 12 primes are always mutually compatible.
    # We need to find 6 of the 14 non-sqfree composites that can all be added.
    # C(14,6) = 3003 combinations - very manageable.
    # For each combo, start with prime_prods and add the 6 composites.

    best_result = list(primes)
    best_count = 12

    # Try all C(14,6) combinations
    for combo in combinations(non_sqfree, 6):
        if time.time() > _DEADLINE:
            break
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
            return list(primes) + list(combo)

    # If no 6-combo works, try 5
    for combo in combinations(non_sqfree, 5):
        if time.time() > _DEADLINE:
            break
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
            if len(primes) + len(combo) > best_count:
                best_count = len(primes) + len(combo)
                best_result = list(primes) + list(combo)

    if time.time() > _DEADLINE:
        return best_result

    # Try 4
    for combo in combinations(non_sqfree, 4):
        if time.time() > _DEADLINE:
            break
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
            if len(primes) + len(combo) > best_count:
                best_count = len(primes) + len(combo)
                best_result = list(primes) + list(combo)

    if time.time() > _DEADLINE:
        return best_result

    # Try 3
    for combo in combinations(non_sqfree, 3):
        if time.time() > _DEADLINE:
            break
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
            if len(primes) + len(combo) > best_count:
                best_count = len(primes) + len(combo)
                best_result = list(primes) + list(combo)

    if time.time() > _DEADLINE:
        return best_result

    # Try 2
    for combo in combinations(non_sqfree, 2):
        if time.time() > _DEADLINE:
            break
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
            if len(primes) + len(combo) > best_count:
                best_count = len(primes) + len(combo)
                best_result = list(primes) + list(combo)

    if time.time() > _DEADLINE:
        return best_result

    # Try 1
    for combo in combinations(non_sqfree, 1):
        if time.time() > _DEADLINE:
            break
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
            if len(primes) + len(combo) > best_count:
                best_count = len(primes) + len(combo)
                best_result = list(primes) + list(combo)

    return best_result if best_count >= 1 else [2]
