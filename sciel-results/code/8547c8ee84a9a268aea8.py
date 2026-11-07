import time
import random
import sys

START_TIME = time.time()
TIME_LIMIT = 85.0

def subset_sums(nums):
    """Generate all nonempty subset sums of nums."""
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
        
    sum_set = set(sums)
    max_sum = max(sum_set)
    
    # Check divisibility
    for s in sum_set:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def can_add_incremental(existing_sums, new_elem):
    """Check if adding new_elem to a set with existing_sums keeps it valid.
    Returns (new_sums, is_valid) where new_sums includes the new element's contributions."""
    new_sums = set(existing_sums)
    for s in existing_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    # Check divisibility among all sums
    sum_list = sorted(new_sums)
    for i, s in enumerate(sum_list):
        # Check if any multiple of s is in the set
        multiple = 2 * s
        for j in range(i + 1, len(sum_list)):
            if sum_list[j] == multiple:
                return None, False
            elif sum_list[j] > multiple:
                break
            multiple += s
    return new_sums, True

def greedy_construct(pool, order, time_limit=85.0):
    """Greedy construction trying to add elements in the given order."""
    current_set = []
    current_sums = set()
    
    for x in order:
        if time.time() - START_TIME > time_limit - 5:
            break
        if x in current_set:
            continue
        new_sums, valid = can_add_incremental(current_sums, x)
        if valid:
            current_set.append(x)
            current_sums = new_sums
    
    return current_set

def construct():
    """
    Returns a list of distinct integers in 1..256 maximizing size,
    such that no two distinct nonempty subset sums divide one another.
    """
    best_set = []
    best_size = 0
    pool = list(range(1, 257))
    
    rng = random.Random(42)
    
    # Phase 1: Try many greedy constructions with different orderings
    # Different orderings can lead to different valid sets
    orderings = []
    
    # Random orderings
    for _ in range(200):
        orderings.append(rng.sample(pool, len(pool)))
    
    # Sorted descending (larger elements first)
    orderings.append(sorted(pool, reverse=True))
    # Sorted ascending
    orderings.append(sorted(pool))
    # Odd numbers first, then even
    orderings.append([x for x in pool if x % 2 == 1] + [x for x in pool if x % 2 == 0])
    # Primes first (approximate)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    primes = [x for x in pool if is_prime(x)]
    non_primes = [x for x in pool if not is_prime(x)]
    orderings.append(primes + non_primes)
    # Multiples of 3 first
    orderings.append([x for x in pool if x % 3 == 0] + [x for x in pool if x % 3 != 0])
    
    for order in orderings:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        result = greedy_construct(pool, order)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Phase 2: Local search from best found
    if best_set:
        current_set = list(best_set)
        current_sums = subset_sums(current_set)
        
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT - 5:
            improved = False
            
            # Try to add any element from pool
            for x in pool:
                if x in current_set:
                    continue
                new_sums, valid = can_add_incremental(current_sums, x)
                if valid:
                    current_set.append(x)
                    current_sums = new_sums
                    improved = True
                    break
            
            # If no addition, try swaps
            if not improved and len(current_set) > 0:
                for i in range(len(current_set)):
                    if time.time() - START_TIME > TIME_LIMIT - 5:
                        break
                    old = current_set[i]
                    for x in pool:
                        if x == old or x in current_set:
                            continue
                        # Try swap
                        temp_set = current_set[:i] + [x] + current_set[i+1:]
                        temp_sums = subset_sums(temp_set)
                        # Quick check: just verify validity
                        sum_list = sorted(temp_sums)
                        valid = True
                        for j, s in enumerate(sum_list):
                            multiple = 2 * s
                            for k in range(j + 1, len(sum_list)):
                                if sum_list[k] == multiple:
                                    valid = False
                                    break
                                elif sum_list[k] > multiple:
                                    break
                                multiple += s
                            if not valid:
                                break
                        if valid:
                            current_set = temp_set
                            current_sums = temp_sums
                            improved = True
                            break
    
    best_set = current_set if current_set else best_set
    
    # Final polish: try to add any remaining element
    final_set = list(best_set)
    final_sums = subset_sums(final_set) if final_set else set()
    for x in pool:
        if x in final_set:
            continue
        new_sums, valid = can_add_incremental(final_sums, x)
        if valid:
            final_set.append(x)
            final_sums = new_sums
    
    return [int(x) for x in final_set]
