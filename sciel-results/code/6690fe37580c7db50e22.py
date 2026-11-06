"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 140

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(s):
        s_set = set(s)
        for a in s:
            for b in s:
                if a + b in s_set:
                    return False
        return True

    def can_add(subset_set, x):
        """Check if x can be added to subset_set while maintaining sum-free."""
        # Need: for all a in subset_set, a+x not in subset_set
        # And: 2x not in subset_set
        # And: x not in subset_set (already checked by caller)
        if x + x in subset_set:
            return False
        for a in subset_set:
            if a + x in subset_set:
                return False
        return True

    def solve_one(arr):
        n = len(arr)
        if n <= 2:
            return list(arr)

        best_subset = None
        best_size = 0

        def greedy(order):
            """Greedy: add elements in given order if sum-free maintained."""
            subset = set()
            for x in order:
                if x in subset:
                    continue
                if can_add(subset, x):
                    subset.add(x)
            return subset

        def local_search(subset):
            """Improve subset via additions and swaps."""
            subset = set(subset)
            remaining = [x for x in arr if x not in subset]
            improved = True
            while improved and time.time() < DEADLINE:
                improved = False
                # Try additions
                for x in remaining[:]:
                    if can_add(subset, x):
                        subset.add(x)
                        remaining.remove(x)
                        improved = True
                # Try swaps: remove y, add x
                for x in remaining[:]:
                    found = False
                    for y in list(subset):
                        new_subset = subset - {y}
                        if can_add(new_subset, x):
                            subset = new_subset | {x}
                            remaining.remove(x)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            return subset

        # Generate candidate orderings
        candidates = []

        # 1. Top half by value
        sorted_asc = sorted(arr)
        med_val = sorted_asc[n // 2]
        cand = [x for x in arr if x > med_val]
        if len(cand) < n // 4:
            cand = [x for x in arr if x >= med_val]
        candidates.append(cand)

        # 2. Bottom half by value
        sorted_desc = sorted(arr, reverse=True)
        med_val2 = sorted_desc[n // 2]
        cand2 = [x for x in arr if x < med_val2]
        if len(cand2) < n // 4:
            cand2 = [x for x in arr if x <= med_val2]
        candidates.append(cand2)

        # 3. Greedy by abs descending
        arr_abs_desc = sorted(arr, key=lambda x: -abs(x))
        candidates.append(list(arr_abs_desc))

        # 4. Greedy by value descending
        candidates.append(sorted(arr, reverse=True))

        # 5. Greedy by value ascending
        candidates.append(sorted(arr))

        # 6. Random orderings
        rng = random.Random(12345)
        for _ in range(30):
            shuffled = arr[:]
            rng.shuffle(shuffled)
            candidates.append(shuffled)

        # Process each candidate
        for i, cand in enumerate(candidates):
            if time.time() > DEADLINE:
                break
            # Greedily build from this ordering
            subset = greedy(cand)
            # Local search
            subset = local_search(subset)
            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = list(subset)

        if best_subset is None:
            best_subset = [max(arr)]

        return best_subset

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            results.append([])
            continue
        arr = instance(size, seed)
        subset = solve_one(arr)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
