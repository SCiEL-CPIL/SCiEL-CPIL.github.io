import time
import random
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 82.0


def _factorize(n, primes):
    """Return exponent vector of n with respect to primes list."""
    vec = []
    for p in primes:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        vec.append(e)
    return tuple(vec)


def _subset_prod_set(primes):
    """Build the set of all 2^k subset products of the given primes."""
    prods = {1}
    for p in primes:
        prods = prods | {q * p for q in prods}
    return prods


def construct():
    if time.time() > _DEADLINE:
        return [2]

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    nprimes = len(primes)

    # Precompute the set of all 2^12 = 4096 subset products of the base primes
    base_prods = _subset_prod_set(primes)

    # Factorize all candidates 2..40
    all_vecs = {}
    for n in range(2, 41):
        all_vecs[n] = _factorize(n, primes)

    candidates = [n for n in range(2, 41) if n not in primes]
    candidate_vecs = [all_vecs[n] for n in candidates]
    n_cand = len(candidates)

    best_count = nprimes
    best_selection = list(primes)

    def _can_add(prods, x):
        """Check if x can be added to the current set of subset products prods.
        Returns True if all new products (q*x for q in prods) are distinct from
        existing prods and from each other."""
        new_prods = set()
        for q in prods:
            r = q * x
            if r in prods or r in new_prods:
                return False
            new_prods.add(r)
        return True

    def _add_to_prods(prods, x):
        """Return new prods set with x multiplied in."""
        return prods | {q * x for q in prods}

    def _search(start_idx, prods, selected_nums, order):
        nonlocal best_count, best_selection
        if time.time() > _DEADLINE - 2:
            return
        cur_count = nprimes + len(selected_nums)
        if cur_count > best_count:
            best_count = cur_count
            best_selection = list(primes) + list(selected_nums)

        for i in range(start_idx, n_cand):
            if time.time() > _DEADLINE - 2:
                return
            idx = order[i]
            x = candidates[idx]
            if _can_add(prods, x):
                new_prods = _add_to_prods(prods, x)
                selected_nums.append(x)
                _search(i + 1, new_prods, selected_nums, order)
                selected_nums.pop()

    # Try multiple orderings via random restarts
    base_order = list(range(n_cand))
    rng = random.Random(42)

    # First pass: deterministic ordering by value
    _search(0, base_prods, [], base_order)

    # Random restarts with shuffled orderings
    remaining_time = _DEADLINE - time.time()
    max_restarts = int(remaining_time / 2.0)  # ~2s per restart
    for r in range(max_restarts):
        if time.time() > _DEADLINE - 3:
            break
        order = base_order[:]
        rng.shuffle(order)
        _search(0, base_prods, [], order)

    result = best_selection[:best_count]
    return [int(x) for x in result]
