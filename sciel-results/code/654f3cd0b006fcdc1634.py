"""Search-based maximization of a divisibility-incomparable subset-sum set."""
import time
import random
from math import gcd
from collections import Counter

DEADLINE_SECONDS = 85.0


def _divides(a, b):
    """True if a divides b (a,b positive ints, a <= b)."""
    if a == b:
        return False
    return b % a == 0


def _has_divisibility_conflict(sums):
    """
    Given a set of positive ints (subset sums), return True if there exist
    two distinct values a < b with a | b.
    Uses multiples-based check: for each a in sums, check if any multiple of a
    (2a, 3a, ...) is also in sums. This is O(|S| * max_sum/a) which is faster
    than O(|S|^2) for sparse sets.
    """
    if len(sums) < 2:
        return False
    arr = sorted(sums)
    n = len(arr)
    max_val = arr[-1]
    for i in range(n):
        a = arr[i]
        # Check multiples of a: 2a, 3a, ... up to max_val
        # For efficiency, only check multiples >= 2a
        multiple = 2 * a
        while multiple <= max_val:
            if multiple in sums:
                return True
            multiple += a
        # Early exit: if 2a > max_val, no larger a can have a multiple in range
        if 2 * a > max_val:
            break
    return False


def _can_add(current_sums, x):
    """
    Check if adding element x to a set with nonempty-subset-sums `current_sums`
    keeps the divisibility-incomparability property.
    New sums = current_sums ∪ {x} ∪ {s + x for s in current_sums}.
    """
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    if _has_divisibility_conflict(new_sums):
        return None
    return new_sums


def _greedy_construct(candidates, start_time, deadline):
    """
    Greedy: try to add candidates in given order, keep the set that grows largest.
    Returns (set_of_elements, set_of_sums).
    """
    elements = []
    sums = set()
    for x in candidates:
        if time.time() > deadline:
            break
        new_sums = _can_add(sums, x)
        if new_sums is not None:
            elements.append(x)
            sums = new_sums
    return elements, sums


def _backtrack_search(pool, current_elements, current_sums, start_time, deadline, best_result):
    """
    Backtracking search with forward-checking.
    Tries to extend current_elements by adding elements from pool.
    Updates best_result in-place if a larger valid set is found.
    """
    if time.time() > deadline:
        return
    
    # If we have a larger set, update best
    if len(current_elements) > len(best_result[0]):
        best_result[0] = list(current_elements)
        best_result[1] = set(current_sums)
    
    # Try each remaining candidate
    for i, x in enumerate(pool):
        if time.time() > deadline:
            return
        
        new_sums = _can_add(current_sums, x)
        if new_sums is not None:
            # Recurse with x added
            _backtrack_search(
                pool[i+1:],  # Only consider candidates after i to avoid duplicates
                current_elements + [x],
                new_sums,
                start_time,
                deadline,
                best_result
            )
            # Continue trying other candidates (don't stop at first success)
            if time.time() > deadline:
                return


def _random_order(pool):
    """Return a random permutation of pool."""
    lst = list(pool)
    random.shuffle(lst)
    return lst


def _validate_solution(elements):
    """
    Validate that the given list of elements satisfies all constraints:
    1. All elements are distinct Python ints in 1..256
    2. No two distinct nonempty subset sums divide each other
    3. No two distinct subsets have the same sum
    Returns True if valid, False otherwise.
    """
    if not isinstance(elements, (list, tuple)):
        return False
    if len(elements) == 0:
        return False
    
    # Check all are Python ints (not bool, not numpy)
    for e in elements:
        if not isinstance(e, int) or isinstance(e, bool):
            return False
        if not (1 <= e <= 256):
            return False
    
    # Check distinctness
    if len(set(elements)) != len(elements):
        return False
    
    # Generate all nonempty subset sums
    n = len(elements)
    all_sums = {}  # sum -> count of subsets that produce it
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        all_sums[s] = all_sums.get(s, 0) + 1
    
    # Check no two distinct subsets have same sum
    for s, cnt in all_sums.items():
        if cnt > 1:
            return False
    
    # Check divisibility incomparability
    sums_list = sorted(all_sums.keys())
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b >= 2 * a and b % a == 0:
                return False
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Candidate pool: integers in 1..256.
    pool = list(range(1, 257))

    best_elements = []
    best_sums = set()

    # First: a deterministic greedy with descending order (larger elements first)
    order = sorted(pool, reverse=True)
    elems, sums = _greedy_construct(order, start_time, deadline)
    if len(elems) > len(best_elements):
        best_elements = elems
        best_sums = sums

    # Also try ascending order
    if time.time() < deadline:
        order = sorted(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Backtracking search with forward-checking
    # Start from a good initial solution and try to extend it
    if time.time() < deadline - 1.0:
        # Use the best greedy result as starting point
        initial_elements = list(best_elements)
        initial_sums = set(best_sums)
        
        # Get remaining candidates
        used = set(initial_elements)
        remaining = [x for x in pool if x not in used]
        
        # Sort remaining in descending order (larger first tends to work better)
        remaining.sort(reverse=True)
        
        best_result = [list(initial_elements), set(initial_sums)]
        _backtrack_search(
            remaining,
            initial_elements,
            initial_sums,
            start_time,
            deadline - 0.5,
            best_result
        )
        
        if len(best_result[0]) > len(best_elements):
            best_elements = best_result[0]
            best_sums = best_result[1]

    # Random restarts with greedy
    max_restarts = 2000
    for r in range(max_restarts):
        if time.time() > deadline - 0.5:
            break
        order = _random_order(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Final: try local improvement on best set by adding remaining elements in random order
    if time.time() < deadline - 0.5:
        used = set(best_elements)
        remaining = [x for x in pool if x not in used]
        random.shuffle(remaining)
        elems = list(best_elements)
        sums = set(best_sums)
        for x in remaining:
            if time.time() > deadline - 0.2:
                break
            new_sums = _can_add(sums, x)
            if new_sums is not None:
                elems.append(x)
                sums = new_sums
        if len(elems) > len(best_elements):
            best_elements = elems

    # Ensure return type is list of Python ints
    return list(best_elements)
