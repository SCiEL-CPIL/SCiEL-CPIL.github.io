"""Model-and-optimize approach for divisibility-incomparable subset sums.

Strategy:
- The constraint is that no two distinct nonempty subset sums divide one another.
- This is a very strong constraint. If we have numbers a < b, then a + b must not be divisible by any other subset sum, and vice versa.
- In particular, if we have two subsets S1 and S2 with sums s1 < s2, we require s2 % s1 != 0.
- Also, all subset sums must be distinct (otherwise equal sums would lead to a division issue as described in the task).

Key insight: Use a set where all subset sums are "large enough" and structured to avoid divisibility.
Use greedy construction with incremental sum tracking, plus local search (swap-based) to improve.

Implementation:
1. Maintain subset sums incrementally for fast validity checks.
2. Greedy search with multiple random orders.
3. Local search: try swapping one element for another to improve set size.
4. Return the best set found within the time budget.
"""

import time
import random

START_TIME = time.time()
TIME_LIMIT = 82.0  # 90s kill, leave 8s margin

def compute_sums(nums):
    """Compute all nonempty subset sums of nums."""
    sums = set()
    n = len(nums)
    if n == 0:
        return sums
    # Use iterative approach
    current = [0]
    for x in nums:
        current += [s + x for s in current]
    sums.update(s for s in current if s > 0)
    return sums

def has_divisibility(sums):
    """Check if any two distinct sums in the set have a divisibility relation."""
    if len(sums) <= 1:
        return False
    sl = sorted(sums)
    n = len(sl)
    for i in range(n):
        si = sl[i]
        # si cannot divide itself, but can divide larger sums
        for j in range(i + 1, n):
            if sl[j] % si == 0:
                return True
    return False

def is_valid(nums):
    """Check if nums forms a valid set."""
    if not nums:
        return True
    sums = compute_sums(nums)
    # All subset sums must be distinct (this is automatic from the set, but we need 
    # to check that no two different subsets give the same sum - actually the set 
    # already handles uniqueness of sum VALUES. The constraint says equal sum values 
    # are counted once, but two different subsets with same sum make the set invalid.
    # So we need to check: are all 2^n - 1 subset sums distinct?
    # Actually, if all subset sums are distinct as values, then no two subsets have 
    # the same sum. Let me verify: the number of distinct subset sums should equal 
    # 2^n - 1 (all nonempty subsets).
    expected_count = (1 << len(nums)) - 1
    if len(sums) != expected_count:
        return False  # duplicate subset sums exist
    return not has_divisibility(sums)

def can_add_fast(current_sums, current_nums, new_elem):
    """Check if adding new_elem keeps the set valid, using current_sums."""
    new_nums = list(current_nums) + [new_elem]
    # New subset sums = old sums + {new_elem} + {s + new_elem for s in old_sums}
    new_sums = set(current_sums)
    new_sums.add(new_elem)
    for s in current_sums:
        new_sums.add(s + new_elem)
    
    # Check distinctness: expected count = 2^(n+1) - 1
    expected_count = (1 << len(new_nums)) - 1
    if len(new_sums) != expected_count:
        return False, None
    
    # Check divisibility
    if has_divisibility(new_sums):
        return False, None
    
    return True, new_sums

def greedy_construct(pool, order_seed=0, time_limit=TIME_LIMIT):
    """Greedy construction: try adding elements in a given order."""
    rng = random.Random(order_seed)
    result = []
    result_set = set()
    current_sums = set()
    remaining = [x for x in pool if x not in result_set]
    rng.shuffle(remaining)
    
    for elem in remaining:
        if time.time() > time_limit:
            break
        ok, new_sums = can_add_fast(current_sums, result, elem)
        if ok:
            result.append(elem)
            result_set.add(elem)
            current_sums = new_sums
            if len(result) >= 30:
                break
    
    return result

def local_search_improve(initial_set, pool, rng, time_limit=TIME_LIMIT, max_swaps=500):
    """Try to improve the set by swapping elements."""
    current = list(initial_set)
    current_set = set(current)
    current_sums = compute_sums(current)
    best = list(current)
    best_size = len(current)
    
    # Build a list of candidates not in current set
    candidates = [x for x in pool if x not in current_set]
    rng.shuffle(candidates)
    
    swaps_done = 0
    for elem_out_idx in range(len(current)):
        if time.time() > time_limit:
            break
        for elem_in in candidates:
            if time.time() > time_limit:
                break
            if elem_in in current_set:
                continue
            
            # Try swapping: remove current[elem_out_idx], add elem_in
            new_set = list(current)
            removed = new_set[elem_out_idx]
            new_set[elem_out_idx] = elem_in
            
            # Check validity
            if is_valid(new_set):
                # This swap maintains validity. Now try to add more elements.
                new_set_set = set(new_set)
                new_sums = compute_sums(new_set)
                
                # Try to greedily add more
                improved = list(new_set)
                improved_set = set(improved)
                improved_sums = new_sums
                remaining_add = [x for x in pool if x not in improved_set]
                rng2 = random.Random(rng.random() * 1000000)
                rng2.shuffle(remaining_add)
                
                for add_elem in remaining_add:
                    if time.time() > time_limit:
                        break
                    ok, new_sums2 = can_add_fast(improved_sums, improved, add_elem)
                    if ok:
                        improved.append(add_elem)
                        improved_set.add(add_elem)
                        improved_sums = new_sums2
                        if len(improved) >= 30:
                            break
                
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
                    current = improved
                    current_set = improved_set
                    current_sums = improved_sums
                    candidates = [x for x in pool if x not in current_set]
                    rng.shuffle(candidates)
                    swaps_done += 1
                    if best_size >= 30:
                        return best
                    break  # restart the outer loop with the improved set
        if time.time() > time_limit:
            break
    
    return best

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    rng_global = random.Random(42)
    
    # Phase 1: Greedy with multiple random orders
    attempts = 0
    while time.time() < TIME_LIMIT - 15:  # leave 15s for local search
        seed = attempts
        result = greedy_construct(pool, order_seed=seed)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
        attempts += 1
        # Also try with pool sorted in various ways
        if attempts % 3 == 0:
            pool_variant = list(pool)
            rng_global.shuffle(pool_variant)
            result2 = greedy_construct(pool_variant, order_seed=seed + 10000)
            if len(result2) > best_size:
                best = result2
                best_size = len(result2)
                if best_size >= 30:
                    break
    
    # Phase 2: Local search to improve
    if best and time.time() < TIME_LIMIT - 5:
        improved = local_search_improve(best, pool, rng_global, time_limit=TIME_LIMIT - 5)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Phase 3: More greedy attempts with different strategies if time remains
    if best_size < 30 and time.time() < TIME_LIMIT - 2:
        # Try starting with larger numbers
        for start_val in [200, 150, 100]:
            if time.time() > TIME_LIMIT - 2:
                break
            pool_high = [x for x in pool if x >= start_val]
            pool_low = [x for x in pool if x < start_val]
            combined = pool_high + pool_low
            rng_global.shuffle(combined)
            result = greedy_construct(combined, order_seed=start_val * 7)
            if len(result) > best_size:
                best = result
                best_size = len(result)
                if best_size >= 30:
                    break
    
    # Ensure we have at least something
    if not best:
        # Find a small valid set
        for a in range(10, 50):
            for b in range(a + 1, 80):
                for c in range(b + 1, 120):
                    if is_valid([a, b, c]):
                        best = [a, b, c]
                        break
                if best:
                    break
            if best:
                break
    
    if not best:
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
