"""Construct a large set of distinct integers in 1..256 whose nonempty subset sums are
pairwise non-dividing (and all distinct).  Uses greedy + local search with a strict
divisibility / distinctness check on the full set of subset sums."""

import random
import time
from itertools import combinations

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _subset_sums_valid(sums_set):
    """Return True iff no two distinct values in sums_set divide one another."""
    vals = sorted(sums_set)
    for i, a in enumerate(vals):
        # any multiple of a (other than a itself) in the set is a violation
        for b in vals[i + 1:]:
            if b % a == 0:
                return False
    return True


def _is_valid(arr):
    """arr: tuple/list of distinct ints.  Check all non-empty subset sums are
    distinct and pairwise non-dividing."""
    n = len(arr)
    sums = set()
    # iterate all non-empty subsets via bit mask
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return _subset_sums_valid(sums)


def _is_valid_incremental(arr, new_elem):
    """Fast check: given a currently-valid arr, test whether adding new_elem
    keeps validity.  We recompute all subset sums of arr ∪ {new_elem} and
    check distinctness + non-divisibility."""
    n = len(arr)
    # existing subset sums
    old_sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        old_sums.add(s)
    # new subset sums: old + new_elem
    new_sums = set(old_sums)
    for s in old_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)

    if len(new_sums) != (1 << (n + 1)) - 1:
        return False
    return _subset_sums_valid(new_sums)


# ---------------------------------------------------------------------------
# greedy construction
# ---------------------------------------------------------------------------

def _greedy_build(pool, target_size, rng):
    """Greedy: add elements from pool (shuffled) that keep validity."""
    arr = []
    candidates = list(pool)
    rng.shuffle(candidates)

    for x in candidates:
        if x in arr:
            continue
        if len(arr) >= target_size:
            break
        if _is_valid_incremental(arr, x):
            arr.append(x)
    return arr


def _greedy_from_seed(seed_elems, pool, target_size, rng):
    arr = list(seed_elems)
    candidates = [x for x in pool if x not in arr]
    rng.shuffle(candidates)
    for x in candidates:
        if len(arr) >= target_size:
            break
        if _is_valid_incremental(arr, x):
            arr.append(x)
    return arr


# ---------------------------------------------------------------------------
# local search (swap)
# ---------------------------------------------------------------------------

def _local_search(arr, pool, target_size, rng, max_iters=2000):
    """Try to swap elements out/in to grow the set."""
    arr = list(arr)
    current_size = len(arr)
    if current_size >= target_size:
        return arr

    for _ in range(max_iters):
        # pick a random element to remove
        i = rng.randrange(len(arr))
        old = arr[i]
        arr2 = arr[:i] + arr[i + 1:]

        # try to add a random new element
        new_candidates = [x for x in pool if x not in arr2 and x != old]
        rng.shuffle(new_candidates)

        improved = False
        for new in new_candidates:
            if _is_valid_incremental(arr2, new):
                arr2.append(new)
                if len(arr2) > current_size:
                    arr = arr2
                    current_size = len(arr)
                    if current_size >= target_size:
                        return arr
                    improved = True
                    break

        if not improved:
            # try a different removal
            continue

    return arr


# ---------------------------------------------------------------------------
# main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 80  # safety margin within 90 s
    rng = random.Random(42)
    pool = list(range(1, 257))

    best = None
    best_size = 0

    # Phase 1: multiple greedy restarts with different orderings
    # Sort pool in various ways to encourage different greedy paths
    strategies = [
        'desc', 'asc', 'random', 'prime_heavy', 'odd_heavy'
    ]

    for strat in strategies:
        if time.time() > deadline - 5:
            break

        pool_ordered = list(pool)
        if strat == 'desc':
            pool_ordered.sort(reverse=True)
        elif strat == 'asc':
            pool_ordered.sort()
        elif strat == 'random':
            rng.shuffle(pool_ordered)
        elif strat == 'prime_heavy':
            # put numbers with few small divisors first (heuristic: larger, co-prime-ish)
            def key(x):
                # count of divisors up to sqrt
                cnt = 0
                for d in range(2, int(x ** 0.5) + 1):
                    if x % d == 0:
                        cnt += 1
                return -x * 100 + cnt  # prefer large, few divisors
            pool_ordered.sort(key=key)
        elif strat == 'odd_heavy':
            # odd numbers tend to have fewer even-sum divisibility issues
            pool_ordered.sort(key=lambda x: (x % 2 == 0, -x))

        # greedy build
        arr = _greedy_build(pool_ordered, 30, rng)
        if len(arr) > best_size:
            best_size = len(arr)
            best = list(arr)

        # local search on the greedy result
        if time.time() > deadline - 10:
            break
        arr2 = _local_search(arr, pool, 30, rng, max_iters=300)
        if len(arr2) > best_size:
            best_size = len(arr2)
            best = list(arr2)

    # Phase 2: more random restarts if time remains
    restarts = 0
    while time.time() < deadline - 10 and restarts < 50:
        restarts += 1
        shuffled = list(pool)
        rng.shuffle(shuffled)
        arr = _greedy_build(shuffled, 30, rng)
        if len(arr) > best_size:
            best_size = len(arr)
            best = list(arr)
        if time.time() > deadline - 10:
            break
        arr2 = _local_search(arr, pool, 30, rng, max_iters=500)
        if len(arr2) > best_size:
            best_size = len(arr2)
            best = list(arr2)

    # Phase 3: targeted local search on the best found
    if best and time.time() < deadline - 5:
        best = _local_search(best, pool, 30, rng, max_iters=5000)
        best_size = len(best)

    if best is None:
        best = [31, 32, 34]

    # Ensure return type is exactly list of Python ints
    return [int(x) for x in best]
