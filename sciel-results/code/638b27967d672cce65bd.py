"""Search-based maximization of a divisibility-incomparable subset-sum set."""
import time
import random
from math import gcd
from collections import Counter

DEADLINE_SECONDS = 82.0


def _divides(a, b):
    """True if a divides b (a,b positive ints, a <= b)."""
    if a == b:
        return False
    return b % a == 0


def _has_divisibility_conflict(sums):
    """
    Given a set of positive ints (subset sums), return True if there exist
    two distinct values a < b with a | b.
    Optimized: for each a, check if any multiple of a (2a, 3a, ...) is in sums.
    """
    if len(sums) < 2:
        return False
    sums_set = set(sums)
    max_sum = max(sums)
    # For each a in sums, check if 2a, 3a, ... are in sums
    for a in sums:
        if a * 2 > max_sum:
            continue
        for k in range(2, max_sum // a + 1):
            if k * a in sums_set:
                return True
    return False


def _can_add(current_sums, x):
    """
    Check if adding element x to a set with nonempty-subset-sums `current_sums`
    keeps the divisibility-incomparability property.
    New sums = current_sums ∪ {x} ∪ {s + x for s in current_sums}.
    Returns new_sums if valid, None otherwise.
    """
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    if _has_divisibility_conflict(new_sums):
        return None
    return new_sums

def _local_search(elements, sums, pool, start_time, deadline):
    """
    Try to improve by swapping elements or adding new ones.
    Returns improved (elements, sums) or original if no improvement.
    """
    best_elements = list(elements)
    best_sums = set(sums)
    used = set(best_elements)
    
    # Try adding remaining elements
    remaining = [x for x in pool if x not in used]
    random.shuffle(remaining)
    for x in remaining:
        if time.time() > deadline - 0.1:
            break
        new_sums = _can_add(best_sums, x)
        if new_sums is not None:
            best_elements.append(x)
            best_sums = new_sums
            used.add(x)
    
    # Try swapping: remove one element, try to add two better ones
    for i in range(len(best_elements)):
        if time.time() > deadline - 0.1:
            break
        removed = best_elements[i]
        temp_elements = best_elements[:i] + best_elements[i+1:]
        # Recompute sums without removed element
        temp_sums = set()
        n = len(temp_elements)
        for mask in range(1, 1 << n):
            s = 0
            for j in range(n):
                if mask & (1 << j):
                    s += temp_elements[j]
            temp_sums.add(s)
        
        # Try to add elements that were previously excluded
        for x in pool:
            if x in used or x == removed:
                continue
            new_sums = _can_add(temp_sums, x)
            if new_sums is not None:
                # Check if we can add another element too
                for y in pool:
                    if y in used or y == removed or y == x:
                        continue
                    new_sums2 = _can_add(new_sums, y)
                    if new_sums2 is not None and len(new_sums2) > len(best_sums):
                        best_elements = temp_elements + [x, y]
                        best_sums = new_sums2
                        used = set(best_elements)
                        break
                else:
                    # Only one element added
                    if len(new_sums) > len(best_sums):
                        best_elements = temp_elements + [x]
                        best_sums = new_sums
                        used = set(best_elements)
                break
    
    return best_elements, best_sums


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

    # First: deterministic greedy with descending order (larger elements first)
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

    # Random restarts
    max_restarts = 3000
    for r in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        order = _random_order(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Local search to improve best solution
    if time.time() < deadline - 0.5:
        improved_elems, improved_sums = _local_search(
            best_elements, best_sums, pool, start_time, deadline
        )
        if len(improved_elems) > len(best_elements):
            best_elements = improved_elems
            best_sums = improved_sums

    # Ensure return type is list of Python ints
    return list(best_elements)
