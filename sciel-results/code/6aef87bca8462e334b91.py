"""Additive bases with few representations – optimized search."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 128
    target_lo, target_hi = 128, 256
    n_targets = target_hi - target_lo + 1  # 129

    # Precompute all pair sums for quick reference
    all_pairs = []
    for i in range(129):
        for j in range(i, 129):
            s = i + j
            if target_lo <= s <= target_hi:
                all_pairs.append((i, j, s))
    all_pairs = np.array(all_pairs, dtype=np.int64)
    pair_i = all_pairs[:, 0]
    pair_j = all_pairs[:, 1]
    pair_s = all_pairs[:, 2] - target_lo  # index into counts array

    def compute_counts_fast(A_set):
        """Compute multiplicity using numpy boolean indexing on precomputed pairs."""
        A_arr = np.array(sorted(A_set), dtype=np.int64)
        in_A = np.zeros(129, dtype=bool)
        in_A[A_arr] = True
        # A pair (i,j) contributes if both i and j are in A
        mask = in_A[pair_i] & in_A[pair_j]
        counts = np.bincount(pair_s[mask], minlength=n_targets)
        return counts

    def compute_counts_fft(A):
        """FFT-based count for smooth objective."""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(n_targets, dtype=np.int64)
        indicator = np.zeros(257, dtype=np.float64)
        indicator[A_arr] = 1.0
        size = 512
        fft_a = np.fft.rfft(indicator, size)
        conv = np.fft.irfft(fft_a * fft_a, size)
        conv = np.round(conv).astype(np.int64)
        targets = np.arange(target_lo, target_hi + 1)
        ordered = conv[targets]
        equal_counts = np.zeros(n_targets, dtype=np.int64)
        even_mask = (targets % 2 == 0)
        if np.any(even_mask):
            half_vals = targets[even_mask] // 2
            A_set = set(A)
            for i, hv in enumerate(half_vals):
                if 0 <= hv <= 128 and hv in A_set:
                    equal_counts[even_mask.nonzero()[0][i]] = 1
        counts = (ordered + equal_counts) // 2
        return counts

    def peak_multiplicity(A):
        counts = compute_counts_fast(set(A))
        return int(counts.max())

    def is_valid(A):
        counts = compute_counts_fast(set(A))
        return bool(np.all(counts > 0))

    def objective(A):
        counts = compute_counts_fast(set(A))
        uncovered = int(np.sum(counts == 0))
        peak = int(counts.max())
        return peak + 100 * uncovered

    def soft_max(counts, beta=3.0):
        c = counts.astype(np.float64)
        c_max = c.max()
        if c_max == 0:
            return 0.0
        shifted = c - c_max
        log_sum = c_max + np.log(np.sum(np.exp(beta * shifted)) / len(c)) / beta
        return log_sum

    def objective_smooth(A):
        counts = compute_counts_fft(A)
        uncovered = int(np.sum(counts == 0))
        peak_smooth = soft_max(counts, beta=3.0)
        return peak_smooth + 100.0 * uncovered

    def smart_init(k):
        """Generate a structured initial set with good coverage properties."""
        # Start with a base that provides good coverage
        # Use arithmetic progression + random perturbations
        base = list(range(0, 129, max(1, 129 // k)))
        random.shuffle(base)
        A = sorted(base[:min(k, len(base))])
        # Fill remaining slots randomly
        remaining = k - len(A)
        if remaining > 0:
            candidates = [x for x in range(129) if x not in A]
            if candidates:
                A.extend(random.sample(candidates, min(remaining, len(candidates))))
        return sorted(set(A))

    def repair(A):
        """Add elements to A to ensure full coverage of [128, 256]."""
        A = sorted(set(A))
        for _ in range(200):
            counts = compute_counts_fast(set(A))
            uncovered = np.where(counts == 0)[0] + target_lo
            if len(uncovered) == 0:
                break
            target = random.choice(list(uncovered))
            best_candidate = None
            best_gain = 0
            for _ in range(30):
                candidate = random.randint(0, 128)
                if candidate not in A:
                    new_A = A + [candidate]
                    new_counts = compute_counts_fast(set(new_A))
                    new_uncovered = int(np.sum(new_counts == 0))
                    old_uncovered = int(np.sum(counts == 0))
                    gain = old_uncovered - new_uncovered
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = candidate
            if best_candidate is not None:
                A = A + [best_candidate]
            else:
                # Try all candidates if random didn't help
                for candidate in range(129):
                    if candidate not in A:
                        new_A = A + [candidate]
                        new_counts = compute_counts_fast(set(new_A))
                        new_uncovered = int(np.sum(new_counts == 0))
                        old_uncovered = int(np.sum(counts == 0))
                        if new_uncovered < old_uncovered:
                            A = new_A
                            break
        return A

    def local_search(A, max_iters=300, use_smooth=False):
        """Improve A with additions, removals, and pair swaps."""
        A = sorted(set(A))
        if use_smooth:
            best_obj = objective_smooth(A)
        else:
            best_obj = objective(A)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try additions
            if not improved:
                for _ in range(20):
                    candidate = random.randint(0, 128)
                    if candidate not in A:
                        new_A = A + [candidate]
                        if use_smooth:
                            new_obj = objective_smooth(new_A)
                        else:
                            new_obj = objective(new_A)
                        if new_obj < best_obj:
                            A = new_A
                            best_obj = new_obj
                            improved = True
                            break

            # Try removals
            if not improved and len(A) > 12:
                for _ in range(15):
                    idx = random.randint(0, len(A) - 1)
                    new_A = A[:idx] + A[idx+1:]
                    if use_smooth:
                        new_obj = objective_smooth(new_A)
                    else:
                        new_obj = objective(new_A)
                    if new_obj < best_obj:
                        A = new_A
                        best_obj = new_obj
                        improved = True
                        break

            # Try pair swaps (remove one, add another)
            if not improved and len(A) >= 2:
                for _ in range(25):
                    idx = random.randint(0, len(A) - 1)
                    candidate = random.randint(0, 128)
                    if candidate not in A:
                        new_A = A[:idx] + [candidate] + A[idx+1:]
                        new_A = sorted(new_A)
                        if use_smooth:
                            new_obj = objective_smooth(new_A)
                        else:
                            new_obj = objective(new_A)
                        if new_obj < best_obj:
                            A = new_A
                            best_obj = new_obj
                            improved = True
                            break

            # Try double swaps (remove two, add two) - helps escape local optima
            if not improved and len(A) >= 4:
                for _ in range(10):
                    idx1, idx2 = random.sample(range(len(A)), 2)
                    c1 = random.randint(0, 128)
                    c2 = random.randint(0, 128)
                    if c1 not in A and c2 not in A and c1 != c2:
                        new_A = [x for i, x in enumerate(A) if i not in (idx1, idx2)]
                        new_A.extend([c1, c2])
                        new_A = sorted(new_A)
                        if use_smooth:
                            new_obj = objective_smooth(new_A)
                        else:
                            new_obj = objective(new_A)
                        if new_obj < best_obj:
                            A = new_A
                            best_obj = new_obj
                            improved = True
                            break

        return A

    def simulated_annealing(A, time_limit):
        """SA using smooth objective for faster evaluation."""
        A = sorted(set(A))
        best_A = A[:]
        best_obj = objective_smooth(A)
        current_obj = best_obj
        T = 12.0
        cooling = 0.999
        min_T = 0.05

        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            if move_type < 0.25 and len(A) > 12:
                idx = random.randint(0, len(A) - 1)
                new_A = A[:idx] + A[idx+1:]
            elif move_type < 0.55:
                candidate = random.randint(0, 128)
                if candidate in A:
                    continue
                new_A = A + [candidate]
            elif move_type < 0.85:
                if len(A) < 1:
                    continue
                idx = random.randint(0, len(A) - 1)
                candidate = random.randint(0, 128)
                if candidate in A:
                    continue
                new_A = A[:idx] + [candidate] + A[idx+1:]
                new_A = sorted(new_A)
            else:
                # Double swap
                if len(A) < 4:
                    continue
                idx1, idx2 = random.sample(range(len(A)), 2)
                c1 = random.randint(0, 128)
                c2 = random.randint(0, 128)
                if c1 in A or c2 in A or c1 == c2:
                    continue
                new_A = [x for i, x in enumerate(A) if i not in (idx1, idx2)]
                new_A.extend([c1, c2])
                new_A = sorted(new_A)

            new_obj = objective_smooth(new_A)
            delta = new_obj - current_obj

            if delta <= 0 or random.random() < np.exp(-delta / T):
                A = new_A
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = A[:]

            T *= cooling

        return best_A, best_obj

    best_A = None
    best_score = float('inf')

    # Try different sizes and restarts
    for k in range(18, 36):
        if time.time() > deadline - 20:
            break

        for restart in range(3):
            if time.time() > deadline - 15:
                break

            A = smart_init(k)
            A = repair(A)
            if not is_valid(A):
                continue

            A = local_search(A, max_iters=200, use_smooth=True)
            obj = objective(A)

            sa_time = min(8, deadline - time.time() - 8)
            if sa_time > 2:
                A, sa_obj = simulated_annealing(A, time.time() + sa_time)
                obj = objective(A)

            if obj < best_score:
                best_score = obj
                best_A = A[:]

    # Final polish with exact objective
    if best_A is not None and time.time() < deadline - 5:
        best_A = local_search(best_A, max_iters=400, use_smooth=False)
        final_obj = objective(best_A)
        if final_obj < best_score:
            best_score = final_obj
            best_A = sorted(best_A)

    if best_A is None or not is_valid(best_A):
        return list(range(64, 129))

    return sorted(best_A)
