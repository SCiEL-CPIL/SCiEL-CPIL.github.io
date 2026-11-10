import time
import random

def construct():
    start = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    N = 256
    universe = list(range(1, N + 1))

    # Precompute conflict masks using bitmasks for speed
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Count conflicts for each number
    conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = conflict_mask[a]
        while mask:
            conflict_count[a] += 1
            mask &= mask - 1

    def count_bits(mask):
        return bin(mask).count('1')

    def is_compatible(x, current_mask):
        return (conflict_mask[x] & current_mask) == 0

    def greedy_add_sorted(order, start_mask):
        current_mask = start_mask
        for e in order:
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
        return current_mask

    def local_search_add(mask):
        """Try to add any compatible element repeatedly"""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not (mask >> x) & 1:
                    if (conflict_mask[x] & mask) == 0:
                        mask |= (1 << x)
                        improved = True
                        break
        return mask

    def local_search_swaps(mask, iters):
        """Try 1-for-1 swaps to improve set size"""
        for _ in range(iters):
            improved = False
            # Try adding any missing element
            for x in range(1, N + 1):
                if not (mask >> x) & 1:
                    if (conflict_mask[x] & mask) == 0:
                        mask |= (1 << x)
                        improved = True
                        break
            if improved:
                continue
            # Try 1-for-1 swap: remove a, add x
            for a in range(1, N + 1):
                if not (mask >> a) & 1:
                    continue
                new_mask = mask & ~(1 << a)
                for x in range(1, N + 1):
                    if (new_mask >> x) & 1:
                        continue
                    if (conflict_mask[x] & new_mask) == 0:
                        mask = new_mask | (1 << x)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return mask

    # Start with odd-only seed (known valid baseline of size 128)
    odds_mask = 0
    for x in range(1, N + 1, 2):
        odds_mask |= (1 << x)

    best_mask = odds_mask
    best_size = count_bits(best_mask)

    # Try to improve with local search on the odd seed
    best_mask = local_search_add(best_mask)
    best_size = count_bits(best_mask)
    best_mask = local_search_swaps(best_mask, 100)
    best_size = count_bits(best_mask)

    if time.time() - start > DEADLINE:
        return [x for x in range(1, N + 1) if (best_mask >> x) & 1]

    # Randomized restarts with different strategies
    evens = list(range(2, N + 1, 2))
    all_nums = list(range(1, N + 1))
    evens_sorted = sorted(evens, key=lambda x: conflict_count[x])
    all_sorted = sorted(all_nums, key=lambda x: conflict_count[x])

    # First: try sorted-by-conflict greedy from odds
    result_mask = greedy_add_sorted(evens_sorted, odds_mask)
    result_mask = local_search_add(result_mask)
    if count_bits(result_mask) > best_size:
        best_size = count_bits(result_mask)
        best_mask = result_mask

    # Try random permutations multiple times
    seed = 0
    while time.time() - start < DEADLINE - 10:
        # Strategy 1: Random greedy from odds
        order = evens[:]
        random.Random(seed).shuffle(order)
        result_mask = greedy_add_sorted(order, odds_mask)
        result_mask = local_search_add(result_mask)
        size = count_bits(result_mask)
        if size > best_size:
            best_size = size
            best_mask = result_mask

        # Strategy 2: Random greedy from empty
        order2 = all_nums[:]
        random.Random(seed + 10000).shuffle(order2)
        result_mask2 = greedy_add_sorted(order2, 0)
        result_mask2 = local_search_add(result_mask2)
        size2 = count_bits(result_mask2)
        if size2 > best_size:
            best_size = size2
            best_mask = result_mask2

        seed += 1

        # Occasionally do more aggressive local search on best
        if seed % 10 == 0 and time.time() - start < DEADLINE - 15:
            temp_mask = best_mask
            temp_mask = local_search_swaps(temp_mask, 50)
            if count_bits(temp_mask) > best_size:
                best_size = count_bits(temp_mask)
                best_mask = temp_mask

    # Final polish: aggressive local search on best
    if time.time() - start < DEADLINE - 5:
        best_mask = local_search_add(best_mask)
        best_mask = local_search_swaps(best_mask, 200)
        best_size = count_bits(best_mask)

    # Convert best_mask back to list
    best = [x for x in range(1, N + 1) if (best_mask >> x) & 1]

    return best
