"""Construct a large dissociated reciprocal set from [1,22]."""
import time
from fractions import Fraction
from itertools import combinations

_DEADLINE = time.time() + 80  # 80s safety margin within 90s kill


def _elapsed():
    return time.time() - _DEADLINE < 0


def _is_dissociated_reciprocal(nums):
    """Return True if all 2^n subset reciprocal sums (incl. empty=0) are distinct."""
    n = len(nums)
    if n == 0:
        return True
    # Represent each element as Fraction(1, x)
    recips = [Fraction(1, x) for x in nums]
    # Incremental subset-sum generation: maintain a set of seen sums
    seen = {Fraction(0, 1)}
    for r in recips:
        # new sums = old sums + r
        new_sums = [s + r for s in seen]
        for ns in new_sums:
            if ns in seen:
                return False
            seen.add(ns)
    return True


def _greedy_with_order(order, deadline_check=None):
    """Greedily add elements in the given order, skipping any that cause a collision."""
    chosen = []
    if deadline_check is not None and deadline_check():
        return chosen
    for x in order:
        candidate = chosen + [x]
        if _is_dissociated_reciprocal(candidate):
            chosen = candidate
        if deadline_check is not None and deadline_check():
            break
    return chosen


def _all_orderings(n=22):
    """Yield several orderings: ascending, descending, random-ish deterministic permutations."""
    base = list(range(1, n + 1))
    yield base
    yield list(reversed(base))
    # A few deterministic "shuffled" orderings via simple LCG
    seed = 12345
    for _ in range(200):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        arr = base[:]
        # Fisher-Yates with LCG
        for i in range(len(arr) - 1, 0, -1):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            j = seed % (i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        yield arr


def _local_search_improve(initial, deadline_check=None):
    """Try to add any remaining element from [1,22] not in initial, one at a time."""
    chosen = list(initial)
    remaining = [x for x in range(1, 23) if x not in chosen]
    improved = True
    while improved and not (deadline_check and deadline_check()):
        improved = False
        for x in remaining:
            candidate = chosen + [x]
            if _is_dissociated_reciprocal(candidate):
                chosen = candidate
                remaining.remove(x)
                improved = True
                if deadline_check and deadline_check():
                    return chosen
    return chosen


def construct():
    best = []
    best_len = 0
    start = time.time()

    for order in _all_orderings(22):
        if time.time() - start > 75:
            break
        result = _greedy_with_order(order, deadline_check=lambda: time.time() - start > 75)
        if len(result) > best_len:
            best = result
            best_len = len(result)

    # Local search improvement on the best found
    if best:
        improved = _local_search_improve(best, deadline_check=lambda: time.time() - start > 78)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)

    # Fallback to seed if something went wrong
    if not best:
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best
