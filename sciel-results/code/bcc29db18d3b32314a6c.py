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
    """
    # Quick pre-check: x must not be divisible by any s, and no s must be divisible by x
    # This is a necessary condition for (x, s) and (s+x, s) and (x, s+x) to be valid.
    # If x % s == 0, then s | x (conflict) or x | (s+x) is false? 
    # Wait. If s | x, then s | x and s | (s+x)? No, s | s+x is always true? No.
    # s | (s+x) iff s | x.
    # So if s | x, then s divides x (conflict) AND s divides s+x (conflict).
    # If x | s, then x divides s (conflict) AND x divides s+x (conflict).
    # So if any s in sums has x%s==0 or s%x==0, we have a conflict.
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
        # This is a random walk on the solution space of fixed size
        if len(elements) > 0:
            # Try a few swaps
            for _ in range(10):
                if time.time() > deadline - 0.5:
                    break
                e_idx = random.randint(0, len(elements) - 1)
                e = elements[e_idx]
                if not remaining:
                    break
                x = random.choice(remaining)
                
                # Try swap: remove e, add x
                # We need to check if (elements - {e} + {x}) is valid
                # This is expensive. 
                # To speed up, we can just check if x can be added to (sums - sums_generated_by_e)?
                # That's hard.
                # Let's just skip swaps for now to keep it simple and fast.
                # Or do it carefully.
                pass

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
    max_restarts = 10000 # Increased
    for r in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Biased random: Prefer larger numbers?
        # Or just random.
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
