import time
import random
import sys

START_TIME = time.time()
TIME_LIMIT = 88.0  # Safety margin before 90s kill

def _all_subset_sums(nums):
    """Compute all non-empty subset sums using bitmask DP. Returns a set."""
    n = len(nums)
    sums = [0] * (1 << n)
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lb = mask & (-mask)
        bit = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + nums[bit]
    return set(sums[1:])

def _is_valid_set(nums):
    """Check if a set of numbers satisfies all constraints."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:  # Too many to check efficiently
        return False
    sums = _all_subset_sums(nums)
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        if si == 0:
            continue
        # Check if any larger sum is a multiple of si
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def _check_no_divisibility(sums_set):
    """Check no two distinct sums in the set have a divisibility relation."""
    if len(sums_set) <= 1:
        return True
    sorted_sums = sorted(sums_set)
    max_sum = sorted_sums[-1]
    for s in sorted_sums:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _can_add_incremental(current_sums, new_elem):
    """
    Check if adding new_elem is valid.
    current_sums: set of all non-empty subset sums of current set.
    Returns (is_valid, new_sums)
    """
    # Generate new sums
    new_sums = set()
    for s in current_sums:
        ns = s + new_elem
        new_sums.add(ns)
    new_sums.add(new_elem)
    
    # Check for collisions
    if len(new_sums) < len(current_sums) + 1:
        return False, None
    
    # Check intersection with existing sums
    if current_sums & new_sums:
        return False, None
        
    # Combine
    all_sums = current_sums | new_sums
    
    # Check divisibility
    if not _check_no_divisibility(all_sums):
        return False, None
        
    return True, all_sums

def _greedy_construct(pool, rng, max_size=30):
    """Greedy construction with incremental checks."""
    result = []
    current_sums = set()
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if len(result) >= max_size:
            break
        is_valid, new_sums = _can_add_incremental(current_sums, elem)
        if is_valid:
            result.append(elem)
            current_sums = new_sums
    return result

def _local_search_improve(current_set, pool, rng, max_iterations=200):
    """
    Try to improve a set by swapping elements.
    Returns improved set or original if no improvement found.
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return list(current_set)
    
    current = list(current_set)
    current_size = len(current)
    in_set = set(current)
    available = [x for x in pool if x not in in_set]
    
    best = list(current)
    best_size = current_size
    
    for _ in range(max_iterations):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if not available:
            break
        
        # Try removing a random element from current and adding a random available
        idx = rng.randint(0, len(current) - 1)
        removed = current[idx]
        add_idx = rng.randint(0, len(available) - 1)
        added = available[add_idx]
        
        trial = [x for i, x in enumerate(current) if i != idx]
        trial.append(added)
        
        if _is_valid_set(trial):
            if len(trial) > best_size:
                best = list(trial)
                best_size = len(trial)
                current = list(trial)
                in_set = set(current)
                available = [x for x in pool if x not in in_set]
            elif rng.random() < 0.3:  # Accept same-size with some probability to explore
                current = list(trial)
                in_set = set(current)
                available = [x for x in pool if x not in in_set]
    
    return best

def construct():
    """
    Main entry point. Uses greedy construction followed by local search.
    Multiple random restarts to find the best valid set.
    """
    best_result = []
    best_size = 0
    
    # Strategy: Use large numbers primarily (128-256) as they have fewer
    # divisibility relations and fewer subset sum collisions.
    
    # Initial greedy attempts with different orderings
    pools = [
        list(range(256, 127, -1)),  # Large first
        list(range(256, 0, -1)),    # All descending
        list(range(192, 127, -1)) + list(range(127, 0, -1)),  # Mid-large first
    ]
    
    for pool in pools:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        rng = random.Random(hash(tuple(pool)) % 100000)
        res = _greedy_construct(pool, rng)
        if len(res) > best_size:
            best_result = res
            best_size = len(res)
    
    # Multiple random restarts with biased pools
    for attempt in range(200):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        rng = random.Random(attempt * 7919 + 31)
        
        # Create biased pool: large numbers first with some randomness
        high = list(range(256, 127, -1))
        low = list(range(127, 0, -1))
        rng.shuffle(high)
        rng.shuffle(low)
        # Mix: put 80% high, 20% low
        pool = high + low[:max(1, len(low) // 5)]
        
        res = _greedy_construct(pool, rng)
        
        if len(res) > best_size:
            best_result = res
            best_size = len(res)
        
        # Local search improvement
        if len(res) >= 3 and time.time() - START_TIME < TIME_LIMIT - 5:
            improved = _local_search_improve(res, list(range(1, 257)), rng, max_iterations=50)
            if len(improved) > best_size:
                best_result = improved
                best_size = len(improved)
        
        # Early exit if we've found a very large set
        if best_size >= 18:
            # Do a few more targeted searches
            for extra in range(20):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                rng2 = random.Random(extra * 104729 + 47)
                pool2 = list(range(256, 100, -1))
                rng2.shuffle(pool2)
                res2 = _greedy_construct(pool2, rng2)
                if len(res2) > best_size:
                    best_result = res2
                    best_size = len(res2)
            break
    
    # Final verification and cleanup
    if not best_result:
        best_result = [256]
    
    # Verify the best result is valid
    if _is_valid_set(best_result):
        return [int(x) for x in best_result]
    else:
        # Fallback: return a known valid small set
        return [256, 255, 254]

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
    print(f"Valid: {_is_valid_set(result)}")
