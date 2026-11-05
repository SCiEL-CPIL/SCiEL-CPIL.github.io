import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

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
        if k <= 50:
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
        else:
            counts = np.zeros(max_sum + 1, dtype=int)
            for i in range(k):
                ai = A_arr[i]
                s = ai + ai
                if s <= max_sum:
                    counts[s] += 1
                for j in range(i + 1, k):
                    aj = A_arr[j]
                    s = ai + aj
                    if s <= max_sum:
                        counts[s] += 1
            return counts

    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, counts)
        return (int(np.max(target_counts)), 0, counts)

    def get_congestion_cost(A):
        """
        Returns a congestion-aware cost that penalizes high multiplicities more strongly.
        Uses sum of squares of target counts as a smooth surrogate for max multiplicity.
        """
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return 10000 + uncovered * 100 + int(np.sum(target_counts ** 2))
        # Primary: max multiplicity, Secondary: sum of squares for smoothness
        max_mult = int(np.max(target_counts))
        sum_sq = int(np.sum(target_counts ** 2))
        return max_mult * 1000 + sum_sq

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

    # Candidate 1: k=16-20, spread out evenly
    for k in [16, 17, 18, 19, 20]:
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

    # Candidate: Two clusters approach
    for a in range(5, 15):
        for b in range(113, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)

    # Candidate: Mix of small and large
    for m in range(6, 12):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid = set(range(60 - m // 2, 60 + m // 2 + 1))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)

    # New: Arithmetic progression with step 2 or 3
    for step in [2, 3]:
        ap_set = set(range(0, 129, step))
        candidates.append(ap_set)

    # New: Symmetric set around 64
    for half in range(8, 16):
        sym_set = set(range(0, half + 1)) | set(range(128 - half, 129))
        candidates.append(sym_set)

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

        # Local search: try swaps and additions/removals with incremental cost
        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)

        for iteration in range(50000):
            if time.time() > deadline - 5.0:
                break

            move_type = rng.random()

            if move_type < 0.6 and len(outside) > 0:
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
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            elif move_type < 0.8 and len(A) > 14:
                # Removal
                x = rng.choice(A_list)

                new_counts, new_cost, _ = incremental_remove_cost(A, x, current_counts)

                if new_cost <= current_cost:
                    A.remove(x)
                    outside.append(x)
                    A_list.remove(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            elif move_type < 1.0 and len(A) < 30:
                # Addition
                x = rng.choice(outside)

                new_counts, new_cost, _ = incremental_add_cost(A, x, current_counts)

                if new_cost <= current_cost:
                    A.add(x)
                    outside.remove(x)
                    A_list.append(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            # Early exit if we reach a very good cost
            if best_cost <= 2:
                break

    # Final polish pass: try to reduce peak multiplicity further
    if best_A is not None and time.time() < deadline - 3.0:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)
        
        # Try targeted swaps to reduce the most congested sums
        for polish_iter in range(2000):
            if time.time() > deadline - 2.0:
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
            for x_in in list(A):
                if found_improvement:
                    break
                # Check if x_in contributes to the congested sum
                partner = max_sum_val - x_in
                if partner in A and partner != x_in:
                    # Try swapping x_in with something else
                    for x_out in outside:
                        new_counts = current_counts.copy()
                        
                        # Remove x_in contributions
                        s = 2 * x_in
                        if s <= max_sum:
                            new_counts[s] -= 1
                        for y in A:
                            if y == x_in:
                                continue
                            s = x_in + y
                            if s <= max_sum:
                                new_counts[s] -= 1
                        
                        # Add x_out contributions
                        s = 2 * x_out
                        if s <= max_sum:
                            new_counts[s] += 1
                        for y in A:
                            if y == x_in:
                                continue
                            s = x_out + y
                            if s <= max_sum:
                                new_counts[s] += 1
                        
                        new_target = new_counts[min_sum:max_sum + 1]
                        new_uncovered = np.sum(new_target == 0)
                        if new_uncovered > 0:
                            continue
                        
                        new_max = int(np.max(new_target))
                        if new_max < max_mult:
                            # Apply the swap
                            A.remove(x_in)
                            A.add(x_out)
                            outside.remove(x_out)
                            outside.append(x_in)
                            current_counts = new_counts
                            current_cost = new_max
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
