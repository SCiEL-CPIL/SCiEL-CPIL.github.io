"""Search-based approach to maximize set size under divisibility constraint on subset sums."""
import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 50.0


def _subset_sums_incr(sums_set, x):
    """Given existing subset sums set (nonempty), add x to all and to itself.
    Returns the new set of subset sums for the augmented list."""
    # Using set comprehension is faster
    return sums_set | {s + x for s in sums_set} | {x}


def _divisibility_conflict(sums_set):
    """Check if any two distinct values in sums_set divide one another.
    Returns True if conflict exists (invalid), False otherwise."""
    if len(sums_set) < 2:
        return False
    vals = sorted(sums_set)
    n = len(vals)
    max_val = vals[-1]
    for i in range(n):
        vi = vals[i]
        # If vi is larger than half the max value, it cannot divide any other distinct value in the set
        # because the smallest multiple would be 2*vi > max_val.
        if vi > max_val // 2:
            continue
        # Check multiples of vi
        # Instead of checking all j, check if k*vi is in the set for k >= 2
        # This is faster if the set is sparse or if we can use set lookup.
        # However, set lookup is O(1). The number of multiples is max_val // vi.
        # In the worst case (vi=1), this is max_val lookups.
        # max_val is sum of elements. If elements are ~100 and size 10, max_val ~1000.
        # 1000 lookups per vi. If n=1000, total 10^6 lookups.
        # This is comparable to O(N^2) but with better constants if vi is small.
        # If vi is large, few lookups.
        
        # Alternative: Loop j from i+1.
        # If we use the 'vi > max_val // 2' skip, we save time for large vi.
        # For small vi, we do many checks.
        # Let's try the multiple lookup approach.
        multiple = vi * 2
        while multiple <= max_val:
            if multiple in sums_set:
                return True
            multiple += vi
    return False


def _is_valid_set(elements):
    """Full validation: all nonempty subset sums, no two distinct divide each other."""
    n = len(elements)
    if n == 0:
        return True
    sums_set = set()
    for x in elements:
        sums_set = _subset_sums_incr(sums_set, x)
    return not _divisibility_conflict(sums_set)


def _greedy_randomized(elements_pool, rng, max_size=30, time_limit=None):
    """Randomized greedy: repeatedly pick a random element that keeps validity."""
    current = []
    sums_set = set()
    remaining = list(elements_pool)
    rng.shuffle(remaining)

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    while len(current) < max_size and remaining:
        # Try elements in random order
        found = False
        for i in range(len(remaining)):
            x = remaining[i]
            # Check if adding x causes divisibility conflict
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                # Accept
                current.append(x)
                sums_set = new_sums
                remaining.pop(i)
                found = True
                break
            if deadline is not None and time.time() > deadline:
                break
        if not found:
            break
        if deadline is not None and time.time() > deadline:
            break

    return current


def _greedy_order_based(elements_pool, order, max_size=30, time_limit=None):
    """Greedy: try elements in given order, add if valid."""
    current = []
    sums_set = set()

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    for x in order:
        if len(current) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        new_sums = _subset_sums_incr(sums_set, x)
        if not _divisibility_conflict(new_sums):
            current.append(x)
            sums_set = new_sums

    return current


def _local_search(current, elements_pool, rng, time_limit=None):
    """Try to improve current set by swapping or adding elements."""
    current = list(current)
    current_set = set(current)
    sums_set = set()
    for x in current:
        sums_set = _subset_sums_incr(sums_set, x)

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    improved = True
    while improved:
        improved = False
        remaining = [x for x in elements_pool if x not in current_set]
        rng.shuffle(remaining)

        # Try adding
        for x in remaining:
            if len(current) >= 30:
                break
            if deadline is not None and time.time() > deadline:
                break
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                current_set.add(x)
                sums_set = new_sums
                improved = True
                break

        if deadline is not None and time.time() > deadline:
            break

        # Try swaps: remove one, add another
        if not improved:
            for i in range(len(current)):
                if deadline is not None and time.time() > deadline:
                    break
                removed = current[i]
                # Remove from sums_set: recompute without removed
                new_current = current[:i] + current[i+1:]
                new_sums = set()
                for x in new_current:
                    new_sums = _subset_sums_incr(new_sums, x)
                for x in remaining:
                    if x in current_set:
                        continue
                    test_sums = _subset_sums_incr(new_sums, x)
                    if not _divisibility_conflict(test_sums):
                        current = new_current + [x]
                        current_set = set(current)
                        sums_set = test_sums
                        improved = True
                        break
                if improved:
                    break

    return current


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Candidate pool: integers in 1..256
    # Bias toward larger numbers (harder to divide)
    # Include a range of sizes
    pool = list(range(1, 257))

    best = []
    best_size = 0

    # Multiple strategies with different orderings
    rng = random.Random(42)

    # Strategy 1: Descending order (larger first - less likely to divide)
    order_desc = list(range(256, 0, -1))
    t_remaining = deadline - time.time()
    if t_remaining > 0.5:
        result = _greedy_order_based(pool, order_desc, max_size=30, time_limit=min(t_remaining - 0.1, 10.0))
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Strategy 2: Randomized greedy with fewer restarts to ensure completion
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
        t_remaining = deadline - time.time() - 1.0
        if t_remaining < 0.5:
            break

        # Create a shuffled pool with bias
        pool_shuffled = list(pool)
        rng.shuffle(pool_shuffled)

        # Bias: try larger numbers first in some runs
        if r % 3 == 0:
            pool_shuffled.sort(reverse=True)
        elif r % 3 == 1:
            # Mix: interleave small and large
            small = [x for x in pool_shuffled if x < 128]
            large = [x for x in pool_shuffled if x >= 128]
            rng.shuffle(small)
            rng.shuffle(large)
            merged = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if si < len(small):
                    merged.append(small[si])
                    si += 1
                if li < len(large):
                    merged.append(large[li])
                    li += 1
            pool_shuffled = merged

        # Limit max size to 20 to ensure speed, we can try to extend later if time permits
        # Actually, just let it run, the deadline will stop it.
        # But to be safe, cap the search to 25 elements.
        result = _greedy_randomized(pool, rng, max_size=25, time_limit=min(t_remaining - 0.5, 2.0))
        if len(result) > best_size:
            best = result
            best_size = len(result)

        # Skip local search to save time, it's very expensive
        # if best_size >= 5 and time.time() < deadline - 2.0:
        #     ls_time = min(deadline - time.time() - 2.0, 3.0)
        #     improved = _local_search(best, pool, rng, time_limit=ls_time)
        #     if len(improved) > best_size:
        #         best = improved
        #         best_size = len(improved)

    # Final: try to extend best set further with a simple greedy pass
    if best and time.time() < deadline - 1.0:
        final_time = min(deadline - time.time() - 1.0, 2.0)
        # Use greedy order based on remaining pool, sorted descending
        remaining_pool = [x for x in pool if x not in set(best)]
        remaining_pool.sort(reverse=True)
        extended = _greedy_order_based(pool, remaining_pool, max_size=30, time_limit=final_time)
        # Merge? No, _greedy_order_based starts from empty.
        # We need to extend `best`.
        # Let's just run a greedy pass on the full pool but pre-populated?
        # The current _greedy_order_based doesn't support pre-populated.
        # I'll write a small extension loop here.
        current = list(best)
        sums_set = set()
        for x in current:
            sums_set = _subset_sums_incr(sums_set, x)
        
        for x in remaining_pool:
            if time.time() > deadline - 0.5:
                break
            if len(current) >= 30:
                break
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                sums_set = new_sums
        
        if len(current) > best_size:
            best = current
            best_size = len(best)

    # Verify best is valid
    if best:
        if not _is_valid_set(best):
            # Fallback to a known small valid set
            best = [31, 32, 34]

    # Ensure types are correct
    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
