"""
Combined approach: constructive number-theoretic interval selection + fast incremental search.

Key structural change from the seed: instead of random restarts over the full 1..256 pool,
we first construct a valid base set by selecting integers from a narrow interval [L, U)
where subset sums of different cardinalities fall into disjoint ranges, reducing divisibility
conflicts. Then we use the fast incremental greedy from the inspirations to extend it.
"""
import time
import random
import math

DEADLINE_SECONDS = 85.0  # Safety margin before 90s limit

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Uses sorted array and multiple-checking for efficiency.
    """
    if len(sums) < 2:
        return False
    
    arr = sorted(sums)
    n = len(arr)
    max_val = arr[-1]
    
    # If smallest * 2 > max, no conflict possible
    if 2 * arr[0] > max_val:
        return False
    
    sum_set = set(arr)
    
    for i in range(n):
        a = arr[i]
        if 2 * a > max_val:
            break
        # Check multiples 2a, 3a, ...
        multiple = 2 * a
        while multiple <= max_val:
            if multiple in sum_set:
                return True
            multiple += a
    
    return False

def _can_add(current_sums_set, x):
    """
    Check if x can be added to a set with nonempty-subset-sums `current_sums_set`.
    Returns new_sums_set or None if invalid.
    
    New sums = current_sums ∪ {x} ∪ {s + x for s in current_sums}
    We need:
    1. x not in current_sums
    2. No s+x collides with any element in current_sums or {x} or other s'+x
    3. No divisibility conflicts in the new set
    """
    # Check x not already a sum
    if x in current_sums_set:
        return None
    
    # Check collisions: s + x must not be in current_sums_set
    # and s + x must not equal x (impossible since s > 0)
    # and s1 + x != s2 + x (impossible since s1 != s2)
    # So we just need s + x not in current_sums_set for all s in current_sums_set
    
    new_sums = set()
    new_sums.add(x)
    for s in current_sums_set:
        ns = s + x
        if ns in current_sums_set:
            return None
        if ns in new_sums:
            return None
        new_sums.add(ns)
    
    # Check divisibility conflict
    if _has_divisibility_conflict(new_sums):
        return None
    
    return new_sums

def _construct_from_interval(L, U, max_size, deadline):
    """
    Construct a set by greedily adding elements from interval [L, U).
    The interval is chosen so that subset sums of different cardinalities
    fall into disjoint ranges, reducing divisibility conflicts.
    
    For elements in [L, U), a subset of size m has sum in [m*L, m*(U-1)].
    These ranges are disjoint if m*(U-1) < (m+1)*L, i.e., U-1 < L + L/m.
    For conservative choice: U - L <= L/2 roughly.
    """
    candidates = list(range(L, min(U, 257)))
    random.shuffle(candidates)
    
    elements = []
    sums_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(elements) >= max_size:
            break
        
        new_sums = _can_add(sums_set, x)
        if new_sums is not None:
            elements.append(x)
            sums_set = new_sums
    
    return elements, sums_set

def _greedy_extend(elements, sums_set, pool, deadline):
    """
    Try to extend an existing set by adding elements from pool.
    """
    used = set(elements)
    remaining = [x for x in pool if x not in used]
    random.shuffle(remaining)
    
    elems = list(elements)
    sset = set(sums_set)
    
    for x in remaining:
        if time.time() > deadline:
            break
        new_sums = _can_add(sset, x)
        if new_sums is not None:
            elems.append(x)
            sset = new_sums
    
    return elems, sset

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    rng = random.Random(12345)
    
    best_elements = []
    best_sums = set()
    best_size = 0
    
    # Strategy 1: Constructive interval-based approach
    # Try various intervals [L, U) where U - L is small relative to L
    # This ensures subset sums of different sizes are in disjoint ranges
    
    intervals = []
    for L in range(2, 128):
        # Choose U such that U - L <= L // 3 (conservative)
        max_U = min(257, L + L // 3 + 1)
        for U in range(L + 1, max_U + 1):
            intervals.append((L, U))
    
    rng.shuffle(intervals)
    
    # Try top intervals
    for L, U in intervals[:200]:
        if time.time() > deadline - 5:
            break
        
        elems, sset = _construct_from_interval(L, U, 30, deadline - 5)
        if len(elems) > best_size:
            best_size = len(elems)
            best_elements = elems
            best_sums = sset
    
    # Strategy 2: Extend best set with random elements from full pool
    if time.time() < deadline - 2 and best_elements:
        pool = list(range(1, 257))
        elems, sset = _greedy_extend(best_elements, best_sums, pool, deadline - 2)
        if len(elems) > best_size:
            best_size = len(elems)
            best_elements = elems
            best_sums = sset
    
    # Strategy 3: Random restarts with full pool (fast greedy)
    for r in range(500):
        if time.time() > deadline - 1:
            break
        
        pool = list(range(1, 257))
        rng.shuffle(pool)
        
        elems = []
        sset = set()
        
        for x in pool:
            if time.time() > deadline - 1:
                break
            new_sums = _can_add(sset, x)
            if new_sums is not None:
                elems.append(x)
                sset = new_sums
        
        if len(elems) > best_size:
            best_size = len(elems)
            best_elements = elems
            best_sums = sset
    
    # Strategy 4: Local improvement on best set
    if time.time() < deadline - 0.5 and best_elements:
        used = set(best_elements)
        remaining = [x for x in range(1, 257) if x not in used]
        rng.shuffle(remaining)
        
        elems = list(best_elements)
        sset = set(best_sums)
        
        for x in remaining:
            if time.time() > deadline - 0.3:
                break
            new_sums = _can_add(sset, x)
            if new_sums is not None:
                elems.append(x)
                sset = new_sums
        
        if len(elems) > best_size:
            best_size = len(elems)
            best_elements = elems
            best_sums = sset
    
    # Final validation: ensure we have a valid set
    if not best_elements:
        # Fallback: try a few specific known-good patterns
        for pattern in [[31, 32, 34], [50, 51, 53, 57], [100, 101, 103, 107, 109]]:
            if _is_valid_set(pattern):
                best_elements = pattern
                break
        else:
            best_elements = [31, 32, 34]
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best_elements]

def _is_valid_set(nums):
    """Validate a set of integers for the task constraints."""
    if not nums:
        return False
    if len(set(nums)) != len(nums):
        return False
    if any(not isinstance(x, int) or isinstance(x, bool) for x in nums):
        return False
    if any(x < 1 or x > 256 for x in nums):
        return False
    
    # Compute all nonempty subset sums
    n = len(nums)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    
    # Check distinctness
    if len(sums) != (1 << n) - 1:
        return False
    
    # Check divisibility
    return not _has_divisibility_conflict(sums)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
    print(f"Valid: {_is_valid_set(result)}")
