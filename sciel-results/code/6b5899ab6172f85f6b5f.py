"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 85.0  # safety margin under 90s


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


def _verify_distinct(selected):
    """Verify all 2^k subset products are distinct. Returns True if valid."""
    prods = {1}
    for x in selected:
        new_items = [p * x for p in prods]
        for item in new_items:
            if item in prods:
                return False
        prods.update(new_items)
    return True


def construct():
    if time.time() > _DEADLINE:
        return [2]

    # All primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Non-square-free composites in 2..40 (candidates to add)
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
    # non_sqfree_composites = [4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28, 32, 36, 40]

    # Start with all primes
    selected = list(primes)
    prods = {1}
    for x in primes:
        _, prods = _try_add(prods, x)

    # Greedily add non-square-free composites in ascending order
    best_selected = selected[:]
    best_prods = prods.copy()

    for x in sorted(non_sqfree_composites):
        if time.time() > _DEADLINE:
            break
        if len(best_selected) >= 18:
            break
        ok, new_prods = _try_add(best_prods, x)
        if ok:
            best_selected.append(x)
            best_prods = new_prods

    if len(best_selected) >= 18:
        # Verify before returning
        if _verify_distinct(best_selected[:18]):
            return list(best_selected[:18])

    # Try descending order
    selected = list(primes)
    prods = {1}
    for x in primes:
        _, prods = _try_add(prods, x)
    for x in sorted(non_sqfree_composites, reverse=True):
        if time.time() > _DEADLINE:
            break
        if len(selected) >= 18:
            break
        ok, new_prods = _try_add(prods, x)
        if ok:
            selected.append(x)
            prods = new_prods
    if len(selected) > len(best_selected):
        best_selected = selected
        best_prods = prods

    if len(best_selected) >= 18:
        if _verify_distinct(best_selected[:18]):
            return list(best_selected[:18])

    # Try all combinations of 6 from the 14 non-sqfree composites
    # C(14,6) = 3003, very manageable
    for combo in combinations(non_sqfree_composites, 6):
        if time.time() > _DEADLINE:
            break
        selected = list(primes) + list(combo)
        if _verify_distinct(selected):
            return selected

    # If we haven't found 18, try to maximize what we have
    # Return best found so far
    if len(best_selected) >= 2:
        return list(best_selected)

    return [2]
# EVOLVE-BLOCK-END
