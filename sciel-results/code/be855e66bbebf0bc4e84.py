import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 86.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129

    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using numpy convolution-like approach"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=int)
        # Compute all pair sums (unordered, with self-pairs)
        A_i = A_arr[:, None]
        A_j = A_arr[None, :]
        sums = A_i + A_j
        mask = np.triu(np.ones((k, k), dtype=bool), k=0)
        valid_sums = sums[mask]
        valid_sums = valid_sums[valid_sums <= max_sum]
        if len(valid_sums) > 0:
            counts = np.bincount(valid_sums, minlength=max_sum + 1)
        else:
            counts = np.zeros(max_sum + 1, dtype=int)
        return counts

    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, counts)
        return (int(np.max(target_counts)), 0, counts)

    def get_smooth_cost(A):
        """
        Returns a smooth cost using log-sum-exp style penalty on high counts.
        This provides a smoother landscape than raw max multiplicity.
        """
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1].astype(float)
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return 10000.0 + uncovered * 100.0 + float(np.sum(target_counts ** 2))
        # Use sum of squares as smooth surrogate + small max penalty
        sum_sq = float(np.sum(target_counts ** 2))
        max_mult = float(np.max(target_counts))
        # Weight: emphasize reducing peaks
        return sum_sq + 10.0 * max_mult * max_mult

    def incremental_swap_cost(A, x_in, x_out, current_counts):
        """
        Compute new counts after removing x_in and adding x_out.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()

        # Subtract contributions involving x_in
        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1

        # Add contributions involving x_out
        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))

        return new_counts, new_cost, uncovered

    def incremental_swap_cost_smooth(A, x_in, x_out, current_counts):
        """
        Compute new counts and smooth cost after removing x_in and adding x_out.
        """
        new_counts = current_counts.copy()

        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1

        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1].astype(float)
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 10000.0 + uncovered * 100.0 + float(np.sum(target_counts ** 2))
        else:
            sum_sq = float(np.sum(target_counts ** 2))
            max_mult = float(np.max(target_counts))
            new_cost = sum_sq + 10.0 * max_mult * max_mult

        return new_counts, new_cost, uncovered

    def incremental_add_cost(A, x_add, current_counts):
        """
        Compute new counts after adding x_add to A.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()

        s = 2 * x_add
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            s = x_add + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))

        return new_counts, new_cost, uncovered

    def incremental_remove_cost(A, x_rem, current_counts):
        """
        Compute new counts after removing x_rem from A.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()

        s = 2 * x_rem
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_rem:
                continue
            s = x_rem + y
            if s <= max_sum:
                new_counts[s] -= 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))

        return new_counts, new_cost, uncovered

    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')

    candidates = []

    # Candidate 1: k=14-20, spread out evenly (smaller sets per cell finding)
    for k in [14, 15, 16, 17, 18, 19, 20]:
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) < k:
            base_set = set()
            used = set()
            for i in range(k):
                x = round(i * 128 / (k - 1))
                while x in used:
                    x += 1
                    if x > 128:
                        x = 0
                base_set.add(x)
                used.add(x)
            base_set = set(x for x in base_set if 0 <= x <= 128)
        candidates.append(set(base_set))

        for offset in range(-3, 4):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128 / (k - 1))) + offset
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 2:
                candidates.append(perturbed)

    # Candidate: Two clusters approach (smaller)
    for a in range(4, 12):
        for b in range(115, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)

    # Candidate: Mix of small and large
    for m in range(5, 10):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid = set(range(60 - m // 2, 60 + m // 2 + 1))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)

    # Arithmetic progression with step 2 or 3
    for step in [2, 3]:
        ap_set = set(range(0, 129, step))
        candidates.append(ap_set)

    # Symmetric set around 64
    for half in range(6, 14):
        sym_set = set(range(0, half + 1)) | set(range(128 - half, 129))
        candidates.append(sym_set)

    # New: Random sparse sets of various sizes
    rng_init = random.Random(42)
    for k in [14, 15, 16, 17, 18]:
        for trial in range(20):
            rand_set = set(rng_init.sample(range(129), k))
            candidates.append(rand_set)

    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost, uncovered, _ = get_cost_detailed(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)

    # Now optimize the best candidate with local search using incremental updates
    if best_A is not None:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)
        current_smooth = get_smooth_cost(A)

        # Local search: try swaps and additions/removals with incremental cost
        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)

        for iteration in range(80000):
            if time.time() > deadline - 4.0:
                break

            move_type = rng.random()

            if move_type < 0.7 and len(outside) > 0:
                # Swap
                x_in = rng.choice(A_list)
                x_out = rng.choice(outside)

                new_counts, new_cost, _ = incremental_swap_cost(A, x_in, x_out, current_counts)

                if new_cost <= current_cost:
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    A_list.remove(x_in)
                    A_list.append(x_out)
                    current_counts = new_counts
                    current_cost = new_cost
                    current_smooth = get_smooth_cost(A)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            elif move_type < 0.85 and len(A) > 12:
                # Removal
                x = rng.choice(A_list)

                new_counts, new_cost, _ = incremental_remove_cost(A, x, current_counts)

                if new_cost <= current_cost:
                    A.remove(x)
                    outside.append(x)
                    A_list.remove(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    current_smooth = get_smooth_cost(A)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            elif move_type < 1.0 and len(A) < 28:
                # Addition
                x = rng.choice(outside)

                new_counts, new_cost, _ = incremental_add_cost(A, x, current_counts)

                if new_cost <= current_cost:
                    A.add(x)
                    outside.remove(x)
                    A_list.append(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    current_smooth = get_smooth_cost(A)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            # Early exit if we reach a very good cost
            if best_cost <= 2:
                break

    # Final polish pass: targeted swaps to reduce peak multiplicity
    if best_A is not None and time.time() < deadline - 2.5:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)
        outside = list(set(range(129)) - A)
        A_list = list(A)

        # Try targeted swaps to reduce the most congested sums
        for polish_iter in range(500):
            if time.time() > deadline - 1.5:
                break

            # Find the sum with highest multiplicity in target range
            target_counts = current_counts[min_sum:max_sum + 1]
            max_idx = int(np.argmax(target_counts))
            max_sum_val = min_sum + max_idx
            max_mult = int(target_counts[max_idx])

            if max_mult <= 2:
                break

            # Try to find a swap that reduces this particular sum's count
            found_improvement = False
            # Only try elements that contribute to the congested sum
            contributing = []
            for x_in in A:
                partner = max_sum_val - x_in
                if partner in A:
                    contributing.append(x_in)

            for x_in in contributing:
                if found_improvement:
                    break
                # Try swapping x_in with each outside element
                for x_out in outside:
                    new_counts, new_cost, _ = incremental_swap_cost(A, x_in, x_out, current_counts)
                    if new_cost < max_mult:
                        A.remove(x_in)
                        A.add(x_out)
                        outside.remove(x_out)
                        outside.append(x_in)
                        A_list.remove(x_in)
                        A_list.append(x_out)
                        current_counts = new_counts
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A)
                        found_improvement = True
                        break

            if not found_improvement:
                break

    # Final verification and cleanup
    if best_A is None:
        # Fallback: return a valid set
        best_A = set(range(0, 129, 8))

    return list(best_A)
