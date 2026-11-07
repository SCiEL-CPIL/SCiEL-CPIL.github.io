import time
import random

# Primes in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PSIZE = len(_PRIMES)

# Precompute prime-exponent vector for each integer 2..40
def _factor_vec(n):
    v = [0] * _PSIZE
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    assert n == 1, f"n={n} not fully factored by primes in [2,40]"
    return tuple(v)

_CANDIDATES = list(range(2, 41))
_VECTORS = {n: _factor_vec(n) for n in _CANDIDATES}

# Deadline guard: keep a safety margin before the 90s kill.
_DEADLINE = time.time() + 80.0


def _subset_products_distinct(vecs):
    """Return True iff all 2^k subset sums of the exponent vectors are distinct.
    Equivalent to all subset products being distinct. Uses integer product set
    for exactness (arbitrary precision)."""
    prods = {1}
    for n in vecs:
        new = {x * n for x in prods}
        if new & prods:
            return False
        prods |= new
    return True


def _can_add(current_prods, n):
    """Check if adding n to a set with current product set keeps all distinct."""
    new = {x * n for x in current_prods}
    if new & current_prods:
        return None
    return new


def _build_products(vecs):
    prods = {1}
    for n in vecs:
        prods |= {x * n for x in prods}
    return prods


def _greedy_build(rng, order):
    """Greedy: iterate order, add if valid."""
    chosen = []
    prods = {1}
    for n in order:
        new = _can_add(prods, n)
        if new is not None:
            chosen.append(n)
            prods = new
    return chosen, prods


def _local_search(chosen, rng, time_limit):
    """Swap-based local search to try to increase |chosen|."""
    chosen = list(chosen)
    prods = _build_products(chosen)
    best = len(chosen)
    while time.time() < time_limit:
        improved = False
        # Try 1-for-1 swaps: remove one, try to add a different one
        for i in range(len(chosen)):
            if time.time() >= time_limit:
                break
            removed = chosen[i]
            # rebuild prods without removed
            prods_wo = _build_products([c for j, c in enumerate(chosen) if j != i])
            for cand in _CANDIDATES:
                if cand in chosen:
                    continue
                new = _can_add(prods_wo, cand)
                if new is not None:
                    # same size, but may enable further growth; accept with some prob
                    # Only accept if it changes the set (cand != removed already ensured)
                    chosen[i] = cand
                    prods = new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try 1-for-2: remove one element, try to add two new ones
            found = False
            for i in range(len(chosen)):
                if time.time() >= time_limit:
                    break
                prods_wo = _build_products([c for j, c in enumerate(chosen) if j != i])
                # try to find two candidates a,b not in chosen such that adding both is valid
                for a in _CANDIDATES:
                    if a in chosen:
                        continue
                    new1 = _can_add(prods_wo, a)
                    if new1 is None:
                        continue
                    for b in _CANDIDATES:
                        if b in chosen or b == a:
                            continue
                        new2 = _can_add(new1, b)
                        if new2 is not None:
                            # success: replace i with a and b
                            new_chosen = [c for j, c in enumerate(chosen) if j != i]
                            new_chosen.append(a)
                            new_chosen.append(b)
                            chosen = new_chosen
                            prods = new2
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                best = max(best, len(chosen))
            else:
                # No growth; do a random perturbation swap to escape local opt
                if len(chosen) >= 2:
                    i = rng.randrange(len(chosen))
                    prods_wo = _build_products([c for j, c in enumerate(chosen) if j != i])
                    cands = [c for c in _CANDIDATES if c not in chosen]
                    if cands:
                        cand = rng.choice(cands)
                        new = _can_add(prods_wo, cand)
                        if new is not None:
                            chosen[i] = cand
                            prods = new
                else:
                    break
    best = max(best, len(chosen))
    return chosen, prods, best


def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    rng = random.Random(12345)
    best_set = []
    best_len = 0

    # Deterministic strong starting set (known valid, 16 elements)
    start = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    if _subset_products_distinct(start):
        best_set = list(start)
        best_len = len(start)

    restarts = 0
    while time.time() < _DEADLINE:
        restarts += 1
        order = list(_CANDIDATES)
        rng.shuffle(order)
        chosen, prods = _greedy_build(rng, order)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = list(chosen)

        # Local search to try to grow
        time_limit = min(_DEADLINE - 2.0, time.time() + 4.0)
        chosen2, prods2, local_best = _local_search(chosen, rng, time_limit)
        if local_best > best_len:
            # verify full validity
            if _subset_products_distinct(chosen2):
                best_len = len(chosen2)
                best_set = list(chosen2)

        # If we already hit a very high count, we can stop early to be safe
        if best_len >= 18:
            # Keep a little more search to try for 19+
            pass

        if restarts > 2000:
            break

    # Final verification of the best set
    if best_set and not _subset_products_distinct(best_set):
        # Fallback to the known-valid 16-element set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]

    return [int(x) for x in best_set]
