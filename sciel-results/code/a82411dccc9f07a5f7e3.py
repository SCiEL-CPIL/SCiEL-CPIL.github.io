"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Represent reciprocal sums as integers scaled by LCM(1..22) for exact
    arithmetic without Fraction overhead.
  - Use a bitmask-free set-based approach with aggressive pruning.
  - Branch-and-bound with multiple orderings and random restarts.
  - Global deadline guard with safety margin.
"""
import time
import random
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 75.0  # wall-clock guard (evaluator kills at 90s; keep 15s margin)
_N = 22

# LCM(1..22) for exact integer representation of reciprocal sums
def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums distinct."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums |= shifted
    return True


def _can_add(sums, v):
    """Check if value v can be added without creating collisions."""
    for s in sums:
        if (v - s) in sums:
            return False
    return True


def _search_order(order, best_ref, deadline, rng):
    """Branch-and-bound with a fixed ordering. best_ref = [card, elements]."""
    n = len(order)
    cur_set = []

    def dfs_inner(idx, sums):
        if time.time() > deadline:
            return
        if len(cur_set) + (n - idx) <= best_ref[0]:
            return
        if idx == n:
            if len(cur_set) > best_ref[0]:
                best_ref[0] = len(cur_set)
                best_ref[1] = list(cur_set)
            return
        
        v = _REC[order[idx]]
        
        # Try include first
        if _can_add(sums, v):
            new_sums = sums | {s + v for s in sums}
            cur_set.append(order[idx])
            dfs_inner(idx + 1, new_sums)
            cur_set.pop()
        
        # Exclude
        dfs_inner(idx + 1, sums)

    dfs_inner(0, {0})


def _verified_baseline():
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return seed if _is_dissociated(seed) else []


def _random_restart_search(best_ref, deadline, rng):
    """Try random orderings to escape local optima."""
    base_order = list(range(1, _N + 1))
    attempts = 0
    while time.time() < deadline and attempts < 50:
        # Shuffle with some structure: keep small numbers earlier
        shuffled = base_order[:]
        rng.shuffle(shuffled)
        # Try this ordering
        _search_order(shuffled, best_ref, deadline, rng)
        attempts += 1


def _greedy_construct(order, deadline):
    """Greedy: add elements if they don't cause collisions."""
    sums = {0}
    result = []
    for e in order:
        v = _REC[e]
        if _can_add(sums, v):
            sums = sums | {s + v for s in sums}
            result.append(e)
        if time.time() > deadline:
            break
    return result


def _local_search(initial, deadline, rng):
    """Hill-climbing: try to add/remove/swap elements to increase size."""
    current = list(initial)
    if not _is_dissociated(current):
        current = _verified_baseline()
    
    best_size = len(current)
    best_set = list(current)
    
    while time.time() < deadline:
        improved = False
        # Try adding elements not in current
        for e in range(1, _N + 1):
            if e not in current:
                candidate = current + [e]
                if _is_dissociated(candidate):
                    current = candidate
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                    improved = True
                    break
        # Try swapping
        if not improved:
            for i in range(len(current)):
                for e in range(1, _N + 1):
                    if e not in current:
                        candidate = current[:i] + current[i+1:] + [e]
                        if _is_dissociated(candidate) and len(candidate) > len(current):
                            current = candidate
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = list(current)
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    return best_set


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    fallback = _verified_baseline()
    best = [len(fallback), list(fallback)]
    deadline = _T0 + _BUDGET
    rng = random.Random(42)

    # Phase 1: Structured orderings
    orderings = [
        list(range(_N, 0, -1)),  # descending
        list(range(1, _N + 1)),  # ascending
        [i for i in range(_N, 0, -2)] + [i for i in range(_N - 1, 0, -2)],
        [19, 17, 13, 11, 7, 5, 3, 2, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 4, 1],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        _search_order(order, best, deadline, rng)

    # Phase 2: Random restarts
    if time.time() < deadline:
        _random_restart_search(best, deadline, rng)

    # Phase 3: Local search from best found
    if time.time() < deadline:
        local_result = _local_search(best[1], deadline, rng)
        if len(local_result) > best[0]:
            best[0] = len(local_result)
            best[1] = local_result

    result = best[1]
    candidates = [c for c in (result, fallback) if c and _is_dissociated(c)]
    if not candidates:
        candidates = [[1]]
    candidates.sort(key=len, reverse=True)
    return candidates[0]
