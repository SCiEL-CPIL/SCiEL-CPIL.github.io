"""Search-based maximization of a divisibility-incomparable subset-sum set."""
import time
import random
from math import gcd

DEADLINE_SECONDS = 85.0

def _has_divisibility_conflict(sums):
    """
    Given a set of positive ints (subset sums), return True if there exist
    two distinct values a < b with a | b.
    """
    if len(sums) < 2:
        return False
    
    max_sum = max(sums)
    
    for a in sums:
        if a * 2 > max_sum:
            continue
        multiple = a * 2
        while multiple <= max_sum:
            if multiple in sums:
                return True
            multiple += a
    return False

def _can_add(current_sums, x):
    """
    Check if adding element x to a set with nonempty-subset-sums `current_sums`
    keeps the divisibility-incomparability property.
    Returns new_sums if valid, else None.
    """
    # Quick pre-check: x must not be divisible by any s, and no s must be divisible by x
    # If s | x, then s | x (conflict) and s | (s+x) (conflict).
    # If x | s, then x | s (conflict) and x | (s+x) (conflict).
    for s in current_sums:
        if x % s == 0 or s % x == 0:
            return None

    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
        
    if _has_divisibility_conflict(new_sums):
        return None
    return new_sums

def _greedy_construct(candidates, start_time, deadline):
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

def _remove_element(elements, sums, idx):
    """
    Remove element at index idx from elements and update sums.
    Returns (new_elements, new_sums) if valid, else (None, None).
    Note: Removing an element might break the 'distinct sums' property?
    No, removing elements reduces the number of subsets.
    However, the remaining set must still satisfy the divisibility constraint.
    Since the original set was valid, any subset of it is also valid?
    Wait. If S is valid, is any subset of S valid?
    Yes. The subset sums of a subset T of S are a subset of the subset sums of S.
    If no two sums in S divide each other, no two sums in T divide each other.
    So removal is always valid.
    """
    e = elements[idx]
    new_elements = elements[:idx] + elements[idx+1:]
    
    # Recalculate sums for the new set?
    # Or update incrementally?
    # Incremental removal is hard. Recalculation is O(2^n).
    # For n up to 30, 2^30 is too slow for frequent calls.
    # But we only do this in local search, and n is likely small-ish or we limit depth.
    # Actually, if n is large, we can't afford O(2^n) per step.
    # However, the current sums set is already computed.
    # Can we update sums by removing e?
    # The new sums are {s in sums | s was not formed by including e?}
    # This is tricky.
    # Let's just recalculate for simplicity if n is small, or use a different strategy.
    # Given the constraint, n is likely < 20 for valid sets found quickly?
    # Let's assume n is manageable for recalculation in local search steps,
    # or we can just use the full validation check which is O(|sums|^2) or similar.
    # Actually, _has_divisibility_conflict is O(|sums| * log(max_sum)).
    # Generating sums is O(2^n * n).
    # If n=15, 2^15 = 32768. Fast.
    # If n=20, 2^20 = 1M. ~0.1s in Python? Maybe.
    # Let's recalculate sums from scratch for the new set.
    
    new_sums = set()
    n = len(new_elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += new_elements[i]
        new_sums.add(s)
        
    # Check validity just in case (should be valid if original was)
    if _has_divisibility_conflict(new_sums):
        return None, None
        
    return new_elements, new_sums

def _local_search(elements, sums, pool, deadline):
    """
    Try to improve the set by adding elements or swapping elements.
    """
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        used = set(elements)
        remaining = [x for x in pool if x not in used]
        
        # 1. Try to add any remaining element
        for x in remaining:
            if time.time() > deadline - 0.5:
                break
            new_sums = _can_add(sums, x)
            if new_sums is not None:
                elements.append(x)
                sums = new_sums
                improved = True
                break # Restart loop with updated set
        
        if improved:
            continue
            
        # 2. If no element could be added, try swapping
        # Pick a random element to remove and a random element to add
        if len(elements) > 0 and remaining:
            # Try a few swaps
            for _ in range(20):
                if time.time() > deadline - 0.5:
                    break
                e_idx = random.randint(0, len(elements) - 1)
                e = elements[e_idx]
                x = random.choice(remaining)
                
                # Try swap: remove e, add x
                new_elements, new_sums_after_remove = _remove_element(elements, sums, e_idx)
                if new_elements is None:
                    continue
                
                # Now try to add x to the set after removal
                candidate_sums = _can_add(new_sums_after_remove, x)
                if candidate_sums is not None:
                    # Valid swap
                    elements = new_elements
                    elements.append(x)
                    sums = candidate_sums
                    improved = True
                    break

    return elements, sums

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    pool = list(range(1, 257))

    best_elements = []
    best_sums = set()

    # Deterministic greedy: Descending
    order = sorted(pool, reverse=True)
    elems, sums = _greedy_construct(order, start_time, deadline)
    if len(elems) > len(best_elements):
        best_elements = elems
        best_sums = sums
        
    # Local search on best
    if time.time() < deadline - 1.0:
        best_elements, best_sums = _local_search(best_elements, best_sums, pool, deadline)

    # Deterministic greedy: Ascending
    if time.time() < deadline - 1.0:
        order = sorted(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums
            if time.time() < deadline - 1.0:
                best_elements, best_sums = _local_search(best_elements, best_sums, pool, deadline)

    # Random restarts
    max_restarts = 10000
    for r in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        order = _random_order(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        
        # Local search on this candidate
        if time.time() < deadline - 0.5:
            elems, sums = _local_search(elems, sums, pool, deadline)
            
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Final polish: Try to add more to best
    if time.time() < deadline - 0.5:
        best_elements, best_sums = _local_search(best_elements, best_sums, pool, deadline)

    return list(best_elements)

def _random_order(pool):
    lst = list(pool)
    random.shuffle(lst)
    return lst
