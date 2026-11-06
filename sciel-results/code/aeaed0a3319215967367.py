"""
Strategy: Exact maximum-size search for reciprocal-subsum-free sets.
The constraint is global (hypergraph), so we use a branch-and-bound DFS with
strong pruning:
  1. Incremental validity check when adding an element (subset-sum with Fractions).
  2. Upper-bound pruning: len(S) + remaining <= best_size.
  3. Time-guarded early termination returning the best valid set found.
We also run a local-search polish (remove-one, add-two) after the main search
to try to squeeze out an extra element if the B&B didn't reach the true optimum
within the time budget.
"""

import time
from fractions import Fraction
from itertools import combinations

_START_TIME = None
_TIME_LIMIT = 82.0  # Leave ~8s buffer for 90s wall-clock limit

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _subset_sum_possible(target, items):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of 'items' (list of Fractions).
    All items are positive.
    Uses DFS with pruning.
    """
    if target == 0:
        return True
    if not items:
        return False
    if target < 0:
        return False

    n = len(items)
    if n == 0:
        return False

    # Sort descending for faster pruning
    items_sorted = sorted(items, key=lambda x: float(x), reverse=True)

    # Suffix sums for pruning
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + items_sorted[i]

    if suffix[0] < target:
        return False

    # Iterative DFS to avoid recursion depth issues
    stack = [(0, Fraction(0))]
    while stack:
        if _time_exceeded():
            return False
        idx, cur = stack.pop()
        if cur == target:
            return True
        if idx >= n:
            continue
        if cur > target:
            continue
        if cur + suffix[idx] < target:
            continue
        # Push skip first (so take is explored first via LIFO)
        stack.append((idx + 1, cur))
        stack.append((idx + 1, cur + items_sorted[idx]))

    return False

def _can_add(x, S):
    """
    Check if integer x can be added to set S (list of ints) without violating constraints.
    S contains distinct integers from 1..24, x not in S.
    """
    if x in S:
        return False

    recip_x = Fraction(1, x)
    recips_S = [Fraction(1, y) for y in S]

    # Check 1: 1/x is not a subset sum of reciprocals of S
    if _subset_sum_possible(recip_x, recips_S):
        return False

    # Check 2: For each y in S with y < x (so 1/y > 1/x), check if (1/y - 1/x)
    # is a subset sum of reciprocals of S \ {y, x}
    for y in S:
        if y < x:
            target = Fraction(1, y) - recip_x
            items = [Fraction(1, z) for z in S if z != y and z != x]
            if _subset_sum_possible(target, items):
                return False
        # If y > x, target would be negative -> impossible with positive items
    return True

def _greedy_initial(order=None):
    """Build a valid set greedily in the given order (default: 1..24)."""
    if order is None:
        order = list(range(1, 25))
    S = []
    for x in order:
        if _can_add(x, S):
            S.append(x)
    return S

def _local_search_polish(S, time_budget=10.0):
    """
    Try to improve S via remove-one-add-two moves.
    Return improved set or original.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    start = time.time()
    S = list(S)
    improved = True
    while improved and (time.time() - start) < time_budget and not _time_exceeded():
        improved = False
        n = len(S)
        # Try removing one element a and adding two elements b, c not in S
        for a in S:
            S_no_a = [x for x in S if x != a]
            candidates = [x for x in range(1, 25) if x not in S]
            for b in candidates:
                if not _can_add(b, S_no_a):
                    continue
                S_with_b = S_no_a + [b]
                for c in candidates:
                    if c == b:
                        continue
                    if not _can_add(c, S_with_b):
                        continue
                    new_S = S_with_b + [c]
                    if len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return S

def construct():
    global _START_TIME
    _START_TIME = time.time()

    # Greedy initialization with a few different orders to get a good lower bound
    best_set = _greedy_initial(list(range(1, 25)))
    # Also try reverse order
    alt = _greedy_initial(list(range(24, 0, -1)))
    if len(alt) > len(best_set):
        best_set = alt
    # Try odd-first
    odds = list(range(1, 24, 2)) + list(range(2, 25, 2))
    alt2 = _greedy_initial(odds)
    if len(alt2) > len(best_set):
        best_set = alt2

    best_size = len(best_set)

    # Branch and bound DFS
    # Order elements so we try to include "safe" ones first.
    # Heuristic: try larger elements first (smaller reciprocals, less likely to form sums)
    # Actually, let's try a mixed order: we want to build a large set quickly.
    # Let's try elements in descending order for the search (24 down to 1).
    # This way, we include large numbers early, which are "safer".

    def search(idx, S):
        nonlocal best_set, best_size
        if _time_exceeded():
            return

        # Upper bound: can we beat best_size?
        remaining = 24 - idx
        if len(S) + remaining <= best_size:
            return

        if idx == 24:
            if len(S) > best_size:
                best_size = len(S)
                best_set = list(S)
            return

        x = 24 - idx  # descending order: 24, 23, ..., 1

        # Try to include x
        if _can_add(x, S):
            S.append(x)
            search(idx + 1, S)
            S.pop()

        # Try to exclude x
        search(idx + 1, S)

    search(0, [])

    # Local search polish on the best set found
    if not _time_exceeded():
        polished = _local_search_polish(best_set, time_budget=8.0)
        if len(polished) > len(best_set):
            best_set = polished

    if not best_set:
        return [24]
    return best_set
