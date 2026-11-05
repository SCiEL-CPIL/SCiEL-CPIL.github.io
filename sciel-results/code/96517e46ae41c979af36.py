import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.array(list(range(min_sum, max_sum + 1)), dtype=np.int64)
    num_targets = len(target_sums)  # 129

    def get_counts_vectorized(A_set):
        """Returns array of counts for sums in [0, 256] using numpy vectorization."""
        A_arr = np.array(sorted(A_set), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=np.int64)
        # Compute all pair sums (unordered, with self-pairs)
        A_i = A_arr[:, None]
        A_j = A_arr[None, :]
        sums = A_i + A_j
        mask = np.triu(np.ones((k, k), dtype=bool))
        valid_sums = sums[mask]
        valid_sums = valid_sums[valid_sums <= max_sum]
        if len(valid_sums) > 0:
            counts = np.bincount(valid_sums, minlength=max_sum + 1).astype(np.int64)
        else:
            counts = np.zeros(max_sum + 1, dtype=np.int64)
        return counts

    def get_target_counts(counts):
        """Extract counts for target range [128, 256]."""
        return counts[min_sum:max_sum + 1]

    def evaluate_set(A_set):
        """Returns (max_mult, uncovered_count, target_counts)."""
        counts = get_counts_vectorized(A_set)
        tc = get_target_counts(counts)
        uncovered = int(np.sum(tc == 0))
        if uncovered > 0:
            return (1000 + uncovered, uncovered, tc)
        max_mult = int(np.max(tc))
        return (max_mult, 0, tc)

    def smooth_objective(tc, t=3.0):
        """
        Smooth surrogate for max multiplicity using log-sum-exp.
        This is a differentiable proxy that encourages uniform distribution.
        """
        # Soft-max via log-sum-exp
        scaled = tc / t
        m = np.max(scaled)
        if m > 500:
            m = 500
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        return lse * t

    # Phase 1: Generate diverse initial candidates
    candidates = []

    # Candidate type 1: Evenly spaced sets of various sizes
    for k in range(14, 25):
        base = set(int(round(i * 128.0 / (k - 1))) for i in range(k))
        base = set(x for x in base if 0 <= x <= 128)
        if len(base) >= k - 2:
            candidates.append(base)
        # Perturbed versions
        for _ in range(3):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128.0 / (k - 1))) + random.randint(-3, 3)
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 3:
                candidates.append(perturbed)

    # Candidate type 2: Two-cluster approach (small + large)
    for a in range(8, 18):
        for b in range(110, 129):
            if b > a + 5:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                if len(clus) >= 10:
                    candidates.append(clus)

    # Candidate type 3: Three-cluster (small, mid, large)
    for m in range(5, 12):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid_start = 55 + random.randint(-5, 5)
        mid_end = 75 + random.randint(-5, 5)
        mid = set(range(max(0, mid_start), min(129, mid_end + 1)))
        mixed = small | large | mid
        if mixed <= set(range(129)) and len(mixed) >= 10:
            candidates.append(mixed)

    # Candidate type 4: Random sets with bias toward coverage
    for _ in range(30):
        k = random.randint(16, 24)
        # Start with some structure
        base = set(random.sample(range(0, 129), k))
        # Ensure we have elements near 0 and 128
        if not any(x < 10 for x in base):
            base.add(random.randint(0, 9))
            if len(base) > k:
                base = set(random.sample(list(base), k))
        if not any(x > 118 for x in base):
            base.add(random.randint(119, 128))
            if len(base) > k:
                base = set(random.sample(list(base), k))
        candidates.append(base)

    # Evaluate all candidates
    best_A = None
    best_max_mult = float('inf')
    best_smooth = float('inf')

    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        max_mult, uncovered, tc = evaluate_set(cand)
        if uncovered > 0:
            continue
        smooth_val = smooth_objective(tc)
        if max_mult < best_max_mult or (max_mult == best_max_mult and smooth_val < best_smooth):
            best_max_mult = max_mult
            best_smooth = smooth_val
            best_A = set(cand)

    if best_A is None:
        # Fallback: find any valid set
        for k in range(12, 30):
            for _ in range(50):
                cand = set(random.sample(range(0, 129), k))
                max_mult, uncovered, tc = evaluate_set(cand)
                if uncovered == 0:
                    best_A = cand
                    best_max_mult = max_mult
                    break
            if best_A is not None:
                break

    if best_A is None:
        return list(range(0, 129, 8))

    # Phase 2: Simulated annealing with vectorized evaluation
    rng = random.Random(42)
    A = set(best_A)
    A_list = list(A)
    outside = list(set(range(129)) - A)
    current_counts = get_counts_vectorized(A)
    current_tc = get_target_counts(current_counts)
    current_max_mult = int(np.max(current_tc)) if np.max(current_tc) > 0 else 0
    current_smooth = smooth_objective(current_tc)

    best_max_mult_sa = current_max_mult
    best_A_sa = set(A)
    best_smooth_sa = current_smooth

    # SA parameters
    T0 = 10.0
    T_final = 0.1
    cooling = 0.9995
    T = T0

    max_iters = 200000
    no_improve = 0
    patience = 5000

    for iteration in range(max_iters):
        if time.time() > deadline - 3.0:
            break

        # Choose move type
        move_type = rng.random()

        if move_type < 0.65 and len(outside) > 0 and len(A) > 0:
            # Swap: remove x_in from A, add x_out
            x_in = rng.choice(A_list)
            x_out = rng.choice(outside)

            # Compute delta in counts
            new_counts = current_counts.copy()

            # Subtract x_in contributions
            s = 2 * x_in
            if s <= max_sum:
                new_counts[s] -= 1
            for y in A_list:
                if y == x_in:
                    continue
                s = x_in + y
                if s <= max_sum:
                    new_counts[s] -= 1

            # Add x_out contributions
            s = 2 * x_out
            if s <= max_sum:
                new_counts[s] += 1
            for y in A_list:
                if y == x_in:
                    continue
                s = x_out + y
                if s <= max_sum:
                    new_counts[s] += 1

            new_tc = get_target_counts(new_counts)
            uncovered = int(np.sum(new_tc == 0))

            if uncovered > 0:
                new_smooth = current_smooth + 100.0 * uncovered
                new_max_mult = current_max_mult + 1000
            else:
                new_max_mult = int(np.max(new_tc))
                new_smooth = smooth_objective(new_tc)

            delta_smooth = new_smooth - current_smooth
            delta_max = new_max_mult - current_max_mult

            # Acceptance: prefer max_mult improvement, use smooth for tie-breaking
            if delta_max < 0:
                accept = True
            elif delta_max == 0:
                if delta_smooth < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta_smooth / T)
            else:
                # Deterioration in max_mult
                if delta_max <= 2 and T > 1.0:
                    accept = rng.random() < np.exp(-delta_max / T)
                else:
                    accept = False

            if accept:
                A.remove(x_in)
                A.add(x_out)
                outside.remove(x_out)
                outside.append(x_in)
                A_list.remove(x_in)
                A_list.append(x_out)
                current_counts = new_counts
                current_tc = new_tc
                current_max_mult = new_max_mult
                current_smooth = new_smooth

                if new_max_mult < best_max_mult_sa or (new_max_mult == best_max_mult_sa and new_smooth < best_smooth_sa):
                    best_max_mult_sa = new_max_mult
                    best_A_sa = set(A)
                    best_smooth_sa = new_smooth
                    no_improve = 0
                else:
                    no_improve += 1

        elif move_type < 0.85 and len(A) > 12:
            # Removal
            x = rng.choice(A_list)

            new_counts = current_counts.copy()
            s = 2 * x
            if s <= max_sum:
                new_counts[s] -= 1
            for y in A_list:
                if y == x:
                    continue
                s = x + y
                if s <= max_sum:
                    new_counts[s] -= 1

            new_tc = get_target_counts(new_counts)
            uncovered = int(np.sum(new_tc == 0))

            if uncovered > 0:
                new_smooth = current_smooth + 100.0 * uncovered
                new_max_mult = current_max_mult + 1000
            else:
                new_max_mult = int(np.max(new_tc))
                new_smooth = smooth_objective(new_tc)

            delta_smooth = new_smooth - current_smooth
            delta_max = new_max_mult - current_max_mult

            if delta_max < 0:
                accept = True
            elif delta_max == 0:
                if delta_smooth < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta_smooth / T)
            else:
                accept = False

            if accept:
                A.remove(x)
                outside.append(x)
                A_list.remove(x)
                current_counts = new_counts
                current_tc = new_tc
                current_max_mult = new_max_mult
                current_smooth = new_smooth

                if new_max_mult < best_max_mult_sa or (new_max_mult == best_max_mult_sa and new_smooth < best_smooth_sa):
                    best_max_mult_sa = new_max_mult
                    best_A_sa = set(A)
                    best_smooth_sa = new_smooth
                    no_improve = 0
                else:
                    no_improve += 1

        elif move_type < 1.0 and len(A) < 35:
            # Addition
            x = rng.choice(outside)

            new_counts = current_counts.copy()
            s = 2 * x
            if s <= max_sum:
                new_counts[s] += 1
            for y in A_list:
                s = x + y
                if s <= max_sum:
                    new_counts[s] += 1

            new_tc = get_target_counts(new_counts)
            uncovered = int(np.sum(new_tc == 0))

            if uncovered > 0:
                new_smooth = current_smooth + 100.0 * uncovered
                new_max_mult = current_max_mult + 1000
            else:
                new_max_mult = int(np.max(new_tc))
                new_smooth = smooth_objective(new_tc)

            delta_smooth = new_smooth - current_smooth
            delta_max = new_max_mult - current_max_mult

            if delta_max < 0:
                accept = True
            elif delta_max == 0:
                if delta_smooth < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta_smooth / T)
            else:
                accept = False

            if accept:
                A.add(x)
                outside.remove(x)
                A_list.append(x)
                current_counts = new_counts
                current_tc = new_tc
                current_max_mult = new_max_mult
                current_smooth = new_smooth

                if new_max_mult < best_max_mult_sa or (new_max_mult == best_max_mult_sa and new_smooth < best_smooth_sa):
                    best_max_mult_sa = new_max_mult
                    best_A_sa = set(A)
                    best_smooth_sa = new_smooth
                    no_improve = 0
                else:
                    no_improve += 1

        # Cool down
        T = max(T_final, T * cooling)

        # Early exit if we've found a very good solution
        if best_max_mult_sa <= 2:
            break

        # Restart if stuck
        if no_improve > patience:
            A = set(best_A_sa)
            A_list = list(A)
            outside = list(set(range(129)) - A)
            current_counts = get_counts_vectorized(A)
            current_tc = get_target_counts(current_counts)
            current_max_mult = int(np.max(current_tc)) if np.max(current_tc) > 0 else 0
            current_smooth = smooth_objective(current_tc)
            no_improve = 0
            T = T0 * 0.5

    # Phase 3: Final local polish - try all single-element swaps
    A_final = set(best_A_sa)
    A_list_final = list(A_final)
    outside_final = list(set(range(129)) - A_final)
    counts_final = get_counts_vectorized(A_final)
    tc_final = get_target_counts(counts_final)
    max_mult_final = int(np.max(tc_final)) if np.max(tc_final) > 0 else 0

    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        A_list_final = list(A_final)
        outside_final = list(set(range(129)) - A_final)
        for x_in in A_list_final:
            for x_out in outside_final:
                if time.time() > deadline - 1.0:
                    break
                new_counts = counts_final.copy()
                # Subtract x_in
                s = 2 * x_in
                if s <= max_sum:
                    new_counts[s] -= 1
                for y in A_final:
                    if y == x_in:
                        continue
                    s = x_in + y
                    if s <= max_sum:
                        new_counts[s] -= 1
                # Add x_out
                s = 2 * x_out
                if s <= max_sum:
                    new_counts[s] += 1
                for y in A_final:
                    if y == x_in:
                        continue
                    s = x_out + y
                    if s <= max_sum:
                        new_counts[s] += 1

                new_tc = get_target_counts(new_counts)
                if np.sum(new_tc == 0) > 0:
                    continue
                new_max = int(np.max(new_tc))
                if new_max < max_mult_final:
                    A_final.remove(x_in)
                    A_final.add(x_out)
                    counts_final = new_counts
                    tc_final = new_tc
                    max_mult_final = new_max
                    improved = True
                    break
            if improved:
                break

    # Final verification
    if len(A_final) == 0:
        A_final = set(range(0, 129, 8))

    final_counts = get_counts_vectorized(A_final)
    final_tc = get_target_counts(final_counts)
    if np.sum(final_tc == 0) > 0:
        # Fallback to best_A_sa if final is invalid
        A_final = best_A_sa

    return list(A_final)
