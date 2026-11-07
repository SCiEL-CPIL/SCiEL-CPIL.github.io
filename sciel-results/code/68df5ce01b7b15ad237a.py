"""Model-and-optimize approach for divisibility-incomparable subset sums.

Strategy:
- The constraint is that no two distinct nonempty subset sums divide one another.
- This is a very strong constraint. If we have numbers a < b, then a + b must not be divisible by any other subset sum, and vice versa.
- In particular, if we have two subsets S1 and S2 with sums s1 < s2, we require s2 % s1 != 0.
- Also, all subset sums must be distinct (otherwise equal sums would lead to a division issue as described in the task).

Key insight: If we use powers of 2 (1, 2, 4, 8, ...), all subset sums are distinct and range from 1 to 2^n - 1. But some will divide others (e.g., 1 divides everything, 2 divides 4, etc.). So powers of 2 don't work.

Better approach: Use a set where all subset sums are "large enough" and structured to avoid divisibility. One known construction is to use numbers that are close together and large, so that subset sums fall in a narrow range where few divisibility relations exist.

However, a simpler and more robust approach is greedy: start with a candidate pool of integers in 1..256, and greedily add elements if they don't violate the constraint. We can try multiple random orders and initial seeds to find a larger set.

Let's implement:
1. A function to check if adding a new element to a set violates the constraint.
2. A greedy search with random restarts and a time deadline.
3. Return the best set found within the time budget.
"""

import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 82.0  # 90s kill, leave 8s margin for safety

def subset_sums(nums):
    """Generate all nonempty subset sums of nums efficiently."""
    sums = {0}
    for x in nums:
        sums = sums | {s + x for s in sums}
    sums.discard(0)
    return sums

def is_valid_set(nums):
    """Check if a set of numbers satisfies the constraint."""
    if not nums:
        return True
    sums = subset_sums(nums)
    if not sums:
        return True
        
    sorted_sums = sorted(sums)
    max_sum = sorted_sums[-1]
    sum_set = set(sums)
    
    for s in sorted_sums:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def can_add(nums, new_elem):
    """Check if adding new_elem to nums keeps the set valid."""
    new_set = list(nums) + [new_elem]
    return is_valid_set(new_set)

def _incremental_validity_check(nums, new_elem):
    """
    Fast check: given a valid set nums, check if adding new_elem keeps it valid.
    We only need to check new subset sums involving new_elem against existing sums.
    """
    if not nums:
        return True
    
    existing_sums = subset_sums(nums)
    new_sums = {new_elem + s for s in existing_sums} | {new_elem}
    
    # Check: no two distinct sums in (existing_sums ∪ new_sums) divide each other
    all_sums = existing_sums | new_sums
    sorted_sums = sorted(all_sums)
    max_sum = sorted_sums[-1]
    sum_set = set(all_sums)
    
    for s in sorted_sums:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def greedy_construct(pool, order_seed=0, bias='random'):
    """Greedy construction: try adding elements in a given order."""
    rng = random.Random(order_seed)
    result = []
    remaining = list(pool)
    
    if bias == 'random':
        rng.shuffle(remaining)
    elif bias == 'desc':
        remaining.sort(reverse=True)
    elif bias == 'asc':
        remaining.sort()
    else:
        rng.shuffle(remaining)

    for elem in remaining:
        if can_add(result, elem):
            result.append(elem)
            if len(result) >= 30:
                break
        if time.time() - START_TIME > TIME_LIMIT:
            break
    return result

def local_search_improve(result, pool, max_swaps=500, seed=0):
    """
    Local search: try swapping elements in result with elements from pool
    to see if we can increase the set size or maintain size while improving.
    """
    if not result:
        return result
    
    rng = random.Random(seed)
    result_set = set(result)
    current = list(result)
    best = list(result)
    best_size = len(result)
    
    # Build candidate pool: elements not in current result
    candidates = [x for x in pool if x not in result_set]
    
    if not candidates:
        return current
    
    # Try multiple swap attempts
    attempts = 0
    max_attempts = max(100, len(candidates) * 2)
    
    while attempts < max_attempts and time.time() - START_TIME < TIME_LIMIT:
        attempts += 1
        
        # Randomly pick an element to remove and an element to add
        if rng.random() < 0.7:
            # Try adding a new element (if we can fit more)
            if len(current) < 30:
                new_elem = rng.choice(candidates)
                trial = current + [new_elem]
                if is_valid_set(trial):
                    current = trial
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
        else:
            # Try swapping: remove one, add one
            if not current or not candidates:
                continue
            remove_idx = rng.randint(0, len(current) - 1)
            new_elem = rng.choice(candidates)
            
            # Don't add an element that's already in the set
            if new_elem in result_set:
                continue
                
            trial = current[:remove_idx] + current[remove_idx+1:] + [new_elem]
            if is_valid_set(trial):
                current = trial
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
    
    return best

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    # Pool: all integers in 1..256
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Phase 1: Greedy construction with multiple orders
    max_greedy_attempts = 1500
    for attempt in range(max_greedy_attempts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Alternate between random, descending, and ascending
        if attempt % 3 == 0:
            bias = 'random'
        elif attempt % 3 == 1:
            bias = 'desc'
        else:
            bias = 'asc'
            
        result = greedy_construct(pool, order_seed=attempt, bias=bias)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
        
    # Phase 2: Local search improvement on the best result found
    if best and time.time() - START_TIME < TIME_LIMIT - 5:
        improved = local_search_improve(best, pool, max_swaps=1000, seed=42)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Phase 3: If we have time, try more greedy + local search combos
    if best_size < 30 and time.time() - START_TIME < TIME_LIMIT - 10:
        for attempt in range(500):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Use different seeds for variety
            seed = 10000 + attempt
            bias = ['random', 'desc', 'asc'][attempt % 3]
            result = greedy_construct(pool, order_seed=seed, bias=bias)
            
            # Apply local search
            if len(result) >= best_size:
                improved = local_search_improve(result, pool, max_swaps=200, seed=seed)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
                    if best_size >= 30:
                        break
    
    # If we didn't find anything good, return a known valid small set
    if not best:
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
