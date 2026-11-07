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
TIME_LIMIT = 85.0  # 90s kill, leave 5s margin

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

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    # Pool: all integers in 1..256
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Try multiple random orders and biases
    max_attempts = 2000
    for attempt in range(max_attempts):
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
        
    # If we didn't find anything good, return a known valid small set
    if not best:
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
