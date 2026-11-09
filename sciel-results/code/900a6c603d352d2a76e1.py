import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit

    max_val = 4096
    max_sum = 3 * max_val

    best_set = []
    best_size = 0

    def try_add_fast(x, arr, used_sums):
        """Try to add x to arr (sorted list), checking all triples involving x."""
        n = len(arr)
        if n == 0:
            used_sums.add(3 * x)
            return True

        # Check (x,x,x)
        s = 3 * x
        if s in used_sums:
            return False

        # Check (x,x,y) and (x,y,y) for y in arr — vectorized
        arr_np = np.array(arr, dtype=np.int32)
        sums_xx_y = 2 * x + arr_np
        sums_x_yy = x + 2 * arr_np
        # Check if any of these are in used_sums
        for s_val in np.unique(sums_xx_y):
            if int(s_val) in used_sums:
                return False
        for s_val in np.unique(sums_x_yy):
            if int(s_val) in used_sums:
                return False

        # Check (x,y,z) for y <= z in arr — vectorized
        # Generate all pairs (i,j) with i <= j
        if n > 1:
            ii, jj = np.triu_indices(n)
            pair_sums = arr_np[ii] + arr_np[jj]
            triple_sums = x + pair_sums
            for s_val in np.unique(triple_sums):
                if int(s_val) in used_sums:
                    return False

        # All checks passed, update used_sums
        used_sums.add(3 * x)
        for s_val in sums_xx_y:
            used_sums.add(int(s_val))
        for s_val in sums_x_yy:
            used_sums.add(int(s_val))
        if n > 1:
            for s_val in triple_sums:
                used_sums.add(int(s_val))

        return True

    def greedy_build(order, deadline):
        """Build a set using greedy approach with given order."""
        current_set = []
        used_sums = set()

        for x in order:
            if time.time() > deadline - 1.0:
                break
            if try_add_fast(x, current_set, used_sums):
                current_set.append(x)

        return current_set

    def build_used_sums(arr):
        """Build the set of all triple sums from a sorted array."""
        used_sums = set()
        n = len(arr)
        arr_np = np.array(arr, dtype=np.int32)
        for i in range(n):
            a = arr_np[i]
            for j in range(i, n):
                b = arr_np[j]
                for k in range(j, n):
                    c = arr_np[k]
                    used_sums.add(int(a + b + c))
        return used_sums

    def local_search(initial_set, deadline, max_iterations=100):
        """Try to improve the set by swapping elements. Uses incremental sum tracking."""
        current_set = sorted(initial_set)
        used_sums = build_used_sums(current_set)

        best_size = len(current_set)
        best_set = current_set[:]

        # Candidate pool: values not in current_set, sorted
        in_set = set(current_set)
        candidates = [v for v in range(1, max_val + 1) if v not in in_set]

        for iteration in range(max_iterations):
            if time.time() > deadline - 1.0:
                break

            improved = False
            n = len(current_set)

            for i in range(n):
                if time.time() > deadline - 1.0:
                    break

                removed = current_set[i]
                # Build temp set without element i
                temp_set = current_set[:i] + current_set[i+1:]
                temp_sums = build_used_sums(temp_set)

                # Try candidates in a smart order: prefer values that minimize max sum
                # Sort candidates by distance from removed to keep density
                # Use a subset of candidates for speed
                if len(candidates) > 200:
                    # Take candidates closest to removed value
                    dist = [abs(c - removed) for c in candidates]
                    # Get top 200 closest
                    idx_sorted = sorted(range(len(candidates)), key=lambda k: dist[k])
                    trial_candidates = [candidates[k] for k in idx_sorted[:200]]
                else:
                    trial_candidates = candidates

                found = False
                for candidate in trial_candidates:
                    if time.time() > deadline - 1.0:
                        break
                    if try_add_fast(candidate, temp_set, temp_sums):
                        # Accept the swap
                        new_set = temp_set + [candidate]
                        new_set.sort()
                        current_set = new_set
                        used_sums = temp_sums
                        # Update used_sums with new element's contributions
                        # (try_add_fast already added them to temp_sums)
                        improved = True
                        found = True
                        break

                if not found:
                    # Restore original
                    pass

            if not improved:
                break

        return current_set

    # Generate diverse ordering strategies
    candidates = list(range(1, max_val + 1))

    orders = []

    # 1. Pure ascending
    orders.append(candidates[:])

    # 2. Descending (large numbers first)
    orders.append(candidates[::-1])

    # 3-6. Modular orderings
    for mod in [3, 5, 7, 11]:
        c_mod = sorted(candidates, key=lambda x: (x % mod, x))
        orders.append(c_mod)

    # 7-10. Random shuffles with fixed seeds for reproducibility
    rng = random.Random(42)
    for _ in range(4):
        c = candidates[:]
        rng.shuffle(c)
        orders.append(c)

    # 11-14. Biased random: favor certain ranges
    for bias_start, bias_end in [(1, 100), (100, 500), (500, 2000), (2000, 4096)]:
        c = candidates[:]
        rng.shuffle(c)
        # Move elements in bias range to front
        biased = [x for x in c if bias_start <= x <= bias_end]
        rest = [x for x in c if not (bias_start <= x <= bias_end)]
        orders.append(biased + rest)

    # Run greedy for each order
    for idx, order in enumerate(orders):
        if time.time() > deadline - 3.0:
            break

        result = greedy_build(order, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]

            # If we have a good set, try local search
            if best_size >= 30 and time.time() < deadline - 8.0:
                improved = local_search(best_set, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]

    return best_set
