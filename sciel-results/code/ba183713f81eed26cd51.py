"""
Program for large reciprocal-subsum-free sets.
Combines:
  - Precomputed global LCM scaling for fast exact subset-sum checks (from inspiration 2)
  - Set-based DP with pruning for subset-sum (faster than MITM for repeated calls)
  - Greedy construction with multiple orderings + random restarts
  - Local-search swap improvement
  - Backtracking search for larger sets
"""
import time
import math
import random
from math import gcd, lcm

_START_TIME = time.time()
_DEADLINE = 80.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


# Precompute global LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i


def _subset_sum_exists(target, elements):
    """
    Check if target can be formed as a sum of a non-empty subset of elements.
    All values are positive integers. Uses set-based DP with pruning.
    """
    if not elements:
        return target == 0
    if target <= 0:
        return False
    if sum(elements) < target:
        return False

    reachable = {0}
    for e in elements:
        new_reachable = set()
        for s in reachable:
            if s <= target:
                new_reachable.add(s)
            ns = s + e
            if ns <= target:
                new_reachable.add(ns)
        reachable = new_reachable

    return target in reachable


def _is_subsum_free_incr(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    selected: list of ints
    candidate: int
    """
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any non-empty subset of selected
    if selected:
        target_int = _R[candidate]
        items_int = [_R[x] for x in selected]
        if _subset_sum_exists(target_int, items_int):
            return False

    # Check for each element in selected: 1/x should not equal sum of reciprocals
    # of any non-empty subset of (selected - {x}) + {candidate}
    for x in selected:
        target_int = _R[x]
        items_int = [_R[y] for y in selected if y != x] + [_R[candidate]]
        if _subset_sum_exists(target_int, items_int):
            return False

    return True


def _is_valid_set(S):
    """
    Verify that S is a valid reciprocal-subsum-free set.
    For each element x in S, check that 1/x is not the sum of reciprocals of
    any non-empty subset of S \ {x}.
    """
    for x in S:
        others = [y for y in S if y != x]
        if not others:
            continue
        target_int = _R[x]
        items_int = [_R[y] for y in others]
        if _subset_sum_exists(target_int, items_int):
            return False
    return True


def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    trial_set = []
    for candidate in order:
        if _time_remaining() < 3.0:
            break
        if _is_subsum_free_incr(trial_set, candidate):
            trial_set.append(candidate)
    return trial_set


def _local_search_improve(S):
    """
    Try to improve set S by removing one element and adding a different one
    that keeps the set valid. Returns the best set found.
    """
    best = list(S)
    best_len = len(best)
    current = set(best)

    # Try all single removals and see if we can add a better element
    for removed in list(best):
        if _time_remaining() < 5.0:
            break
        remaining = [x for x in best if x != removed]

        # Try to add any element not in current set
        for candidate in range(24, 0, -1):
            if candidate in current:
                continue
            if _time_remaining() < 3.0:
                break
            new_set = remaining + [candidate]
            if _is_valid_set(new_set):
                if len(new_set) > best_len:
                    best = new_set
                    best_len = len(best)
                    current = set(best)
                    break

    # Try all single additions
    for candidate in range(24, 0, -1):
        if candidate in current:
            continue
        if _time_remaining() < 3.0:
            break
        new_set = best + [candidate]
        if _is_valid_set(new_set):
            if len(new_set) > best_len:
                best = new_set
                best_len = len(best)
                current = set(best)

    return best


def _backtrack_search(current, order, target_size):
    """
    Backtracking search to find a valid set of size target_size.
    current: list of elements already selected
    order: list of remaining candidates in priority order
    target_size: desired final size
    Returns a valid set of size target_size or None.
    """
    if _time_remaining() < 2.0:
        return None

    if len(current) == target_size:
        if _is_valid_set(current):
            return list(current)
        return None

    # Pruning: not enough candidates left
    if len(current) + len(order) < target_size:
        return None

    for i, candidate in enumerate(order):
        if _time_remaining() < 2.0:
            return None

        new_current = current + [candidate]
        remaining = order[:i] + order[i+1:]

        # Quick check: is new_current valid so far?
        if _is_valid_set(new_current):
            result = _backtrack_search(new_current, remaining, target_size)
            if result is not None:
                return result

    return None


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses greedy construction with multiple orderings, random restarts,
    local search improvement, and backtracking for larger sets.
    """
    all_elements = list(range(1, 25))
    best_set = []

    # 1. Deterministic greedy: Large to Small
    if _time_remaining() > 10.0:
        s = _greedy_construct(sorted(all_elements, reverse=True))
        if len(s) > len(best_set):
            best_set = s

    # 2. Deterministic greedy: Small to Large
    if _time_remaining() > 10.0:
        s = _greedy_construct(sorted(all_elements))
        if len(s) > len(best_set):
            best_set = s

    # 3. Deterministic greedy: Alternating large/small
    if _time_remaining() > 10.0:
        order = []
        large = sorted(all_elements, reverse=True)
        small = sorted(all_elements)
        for i in range(12):
            if i < len(large):
                order.append(large[i])
            if i < len(small):
                order.append(small[i])
        s = _greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s

    # 4. Local search improvement on the best so far
    if _time_remaining() > 15.0 and best_set:
        improved = _local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # 5. Random restarts
    random.seed(42)
    for _ in range(200):
        if _time_remaining() < 15.0:
            break

        candidates = list(all_elements)
        random.shuffle(candidates)
        trial_set = _greedy_construct(candidates)

        if len(trial_set) > len(best_set):
            best_set = trial_set
            if _time_remaining() > 15.0:
                improved = _local_search_improve(best_set)
                if len(improved) > len(best_set):
                    best_set = improved

    # 6. Biased random restarts: favor large elements
    for _ in range(80):
        if _time_remaining() < 15.0:
            break

        candidates = list(all_elements)
        random.shuffle(candidates)
        large_elems = [x for x in candidates if x > 12]
        small_elems = [x for x in candidates if x <= 12]
        random.shuffle(large_elems)
        random.shuffle(small_elems)
        candidates = large_elems + small_elems

        trial_set = _greedy_construct(candidates)

        if len(trial_set) > len(best_set):
            best_set = trial_set
            if _time_remaining() > 15.0:
                improved = _local_search_improve(best_set)
                if len(improved) > len(best_set):
                    best_set = improved

    # 7. Try backtracking for larger sets
    current_best_size = len(best_set)
    for target_size in range(current_best_size + 1, min(current_best_size + 4, 25)):
        if _time_remaining() < 20.0:
            break

        # Try backtracking with different orderings
        for order in [sorted(all_elements, reverse=True), sorted(all_elements)]:
            if _time_remaining() < 10.0:
                break
            result = _backtrack_search([], order, target_size)
            if result is not None:
                best_set = result
                current_best_size = target_size
                break

    # 8. Final local search pass on the best set
    if _time_remaining() > 10.0 and best_set:
        improved = _local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # 9. Try to add any remaining elements
    if _time_remaining() > 5.0 and best_set:
        current_set = set(best_set)
        for candidate in range(24, 0, -1):
            if candidate in current_set:
                continue
            if _time_remaining() < 3.0:
                break
            new_set = best_set + [candidate]
            if _is_valid_set(new_set):
                best_set = new_set
                current_set.add(candidate)

    # Ensure nonempty
    if not best_set:
        best_set = [24]

    return sorted(best_set)


if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
