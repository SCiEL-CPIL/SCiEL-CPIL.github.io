import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    N = 256
    MAX_SIZE = 60

    def check_valid(vals):
        """Check if vals satisfy the cardinality-separated subset sum property.
        Returns True if valid. Uses bitset DP per cardinality."""
        if not vals:
            return True
        n = len(vals)
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        # Max possible sum: n * max(vals)
        max_sum = sum(vals)
        # Use list of ints as bitsets
        # sums[k] has bits for achievable sums with exactly k elements
        sums = [0] * (n + 1)
        sums[0] = 1  # sum 0 with 0 elements
        for v in vals:
            # Update in reverse order of k to avoid using same element twice
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    # Shift sums[k] left by v and OR into sums[k+1]
                    sums[k + 1] |= (sums[k] << v)
        
        # Now check pairwise disjointness of sums[k] for different k
        # For each pair (i, j) with i < j, check sums[i] & sums[j] == 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True

    def try_greedy_descending(pool, max_size):
        """Try to build a valid set by greedily adding elements from largest to smallest."""
        result = []
        for v in sorted(pool, reverse=True):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_greedy_ascending(pool, max_size):
        """Try to build a valid set by greedily adding elements from smallest to largest."""
        result = []
        for v in sorted(pool):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_randomized(pool, max_size, iterations):
        """Try randomized orderings to find larger valid sets."""
        best = []
        for _ in range(iterations):
            shuffled = pool[:]
            random.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= max_size:
                    break
                candidate = result + [v]
                if check_valid(candidate):
                    result = candidate
            if len(result) > len(best):
                best = result
                if len(best) >= max_size:
                    break
        return best

    # Strategy 1: Try descending greedy (large values first tends to separate sums)
    pool = list(range(1, N + 1))
    best = try_greedy_descending(pool, MAX_SIZE)
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Try ascending greedy
    asc = try_greedy_ascending(pool, MAX_SIZE)
    if len(asc) > len(best):
        best = asc
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Try subsets of the pool - maybe not all values are useful
    # Try with only larger values
    for start in range(1, 50):
        if time.time() > deadline - 5:
            break
        sub_pool = list(range(start, N + 1))
        desc = try_greedy_descending(sub_pool, MAX_SIZE)
        if len(desc) > len(best):
            best = desc
    
    if time.time() > deadline:
        return best
    
    # Strategy 4: Randomized attempts
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        # Estimate how many iterations we can afford
        # Each check_valid is O(n^2 * bitset_ops), roughly fast for n up to 60
        # Try a few randomized attempts
        iterations = min(50, int(remaining_time * 20))
        rand_best = try_randomized(pool, MAX_SIZE, iterations)
        if len(rand_best) > len(best):
            best = rand_best
    
    # Ensure we return a list of Python ints
    best = [int(x) for x in best]
    return best
