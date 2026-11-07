import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    def get_subset_sums(S):
        """Get all non-empty subset sums of S."""
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums

    def is_valid(S):
        """Check if all non-empty subset sums are divisibility-incomparable."""
        sums = get_subset_sums(S)
        sums_list = sorted(sums)
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                a, b = sums_list[i], sums_list[j]
                if b % a == 0:
                    return False
        return True

    def can_add(S, x, current_sums):
        """Check if adding x to S keeps the property, given current sums."""
        # New sums are: x, and x + s for each s in current_sums
        new_sums = {x}
        for s in current_sums:
            new_sums.add(x + s)
        # All sums must be divisibility-incomparable
        all_sums = current_sums | new_sums
        sums_list = sorted(all_sums)
        for i in range(len(sums_list)):
            a = sums_list[i]
            for j in range(i + 1, len(sums_list)):
                b = sums_list[j]
                if b % a == 0:
                    return False
        return True

    # Start with a known valid small set and try to grow it
    # Use greedy: try adding candidates in various orders
    best = [11, 13, 17]
    best_size = len(best)

    def try_greedy(initial, candidates, deadline):
        """Greedy extend: add candidates if they fit."""
        S = list(initial)
        current_sums = get_subset_sums(S)
        for x in candidates:
            if x in S:
                continue
            if time.time() > deadline:
                break
            if can_add(S, x, current_sums):
                S.append(x)
                # Update sums
                new_sums = {x}
                for s in current_sums:
                    new_sums.add(x + s)
                current_sums = current_sums | new_sums
        return S, current_sums

    # Candidate pool: numbers that are "large" to avoid divisibility issues
    # Smaller numbers are more likely to divide larger subset sums
    # Try different strategies

    # Strategy 1: Greedy from small primes/large numbers
    # Large numbers are less likely to divide other sums
    all_candidates = list(range(1, 257))

    # Try starting from different seeds
    seed_sets = [
        [127, 129, 131],
        [100, 101, 103],
        [80, 81, 83],
        [60, 61, 67],
        [50, 53, 59],
        [40, 41, 43],
        [30, 31, 37],
        [20, 23, 29],
        [15, 17, 19],
        [11, 13, 17],
        [10, 11, 13],
        [9, 11, 13],
        [8, 11, 13],
        [7, 11, 13],
    ]

    for seed in seed_sets:
        if time.time() > deadline:
            break
        if not is_valid(seed):
            continue
        # Try greedy with candidates sorted in various orders
        # Larger numbers first (less likely to divide)
        candidates_desc = [x for x in all_candidates if x not in seed]
        candidates_desc.sort(reverse=True)
        S, _ = try_greedy(seed, candidates_desc, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)

        # Also try ascending
        candidates_asc = [x for x in all_candidates if x not in seed]
        candidates_asc.sort()
        S, _ = try_greedy(seed, candidates_asc, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Randomized restarts
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        # Random seed of 2-4 elements
        seed_size = random.randint(2, 4)
        seed = random.sample(range(1, 257), seed_size)
        seed.sort()
        if not is_valid(seed):
            continue
        # Random order of candidates
        candidates = [x for x in range(1, 257) if x not in seed]
        random.shuffle(candidates)
        S, _ = try_greedy(seed, candidates, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Local search: try to swap elements to improve
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_sums = get_subset_sums(best)
            for i in range(len(best)):
                if time.time() > deadline - 5.0:
                    break
                for x in range(1, 257):
                    if x in best:
                        continue
                    new_set = best[:i] + [x] + best[i+1:]
                    new_set.sort()
                    if is_valid(new_set):
                        # Check if this swap helps (we want to keep size, but maybe enable future growth)
                        # For now, just accept if valid
                        pass  # Swaps don't increase size, skip for now

    # Final verification
    if not is_valid(best):
        # Fallback to a known valid set
        best = [31, 32, 34]

    return list(best)
