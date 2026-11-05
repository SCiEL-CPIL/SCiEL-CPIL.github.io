"""Additive bases with few representations – optimized search."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s kill

    n = 128
    target_lo, target_hi = 128, 256
    n_targets = target_hi - target_lo + 1  # 129

    # FFT-based count computation: fast for repeated evaluations
    # count_unordered(s) = (ordered_count(s) + count_equal(s)) / 2

    def compute_counts_fft(A):
        """Compute multiplicity of each sum in [128, 256] using FFT."""
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

    def compute_counts_direct(A):
        """Compute multiplicity of each sum in [128, 256] directly."""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(n_targets, dtype=np.int64)

        i_idx, j_idx = np.triu_indices(k, k=0)
        sums = A_arr[i_idx] + A_arr[j_idx]
        mask = (sums >= target_lo) & (sums <= target_hi)
        relevant_sums = sums[mask]
        counts = np.bincount(relevant_sums - target_lo, minlength=n_targets)
        return counts

    def peak_multiplicity(A):
        """Return the maximum representation count in [128, 256]."""
        counts = compute_counts_direct(A)
        return int(counts.max())

    def is_valid(A):
        """Check that every value in [128, 256] is covered."""
        counts = compute_counts_direct(A)
        return bool(np.all(counts > 0))

    def objective(A):
        """Combined objective: peak multiplicity + penalty for uncovered values."""
        counts = compute_counts_direct(A)
        uncovered = int(np.sum(counts == 0))
        peak = int(counts.max())
        return peak + 100 * uncovered

    # Smooth soft-max surrogate for SA
    def soft_max(counts, beta=3.0):
        """Log-sum-exp based soft max."""
        c = counts.astype(np.float64)
        c_max = c.max()
        if c_max == 0:
            return 0.0
        shifted = c - c_max
        log_sum = c_max + np.log(np.sum(np.exp(beta * shifted)) / len(c)) / beta
        return log_sum

    def objective_smooth(A):
        """Smooth objective using FFT counts and soft-max."""
        counts = compute_counts_fft(A)
        uncovered = int(np.sum(counts == 0))
        peak_smooth = soft_max(counts, beta=3.0)
        return peak_smooth + 100.0 * uncovered

    def random_init(k):
        """Generate a random subset of [0,128] of size k."""
        return sorted(random.sample(range(0, 129), k))

    def repair(A):
        """Add elements to A to ensure full coverage of [128, 256]."""
        A = sorted(set(A))
        for _ in range(300):
            counts = compute_counts_direct(A)
            uncovered = np.where(counts == 0)[0] + target_lo
            if len(uncovered) == 0:
                break
            target = random.choice(list(uncovered))
            for _ in range(80):
                candidate = random.randint(0, 128)
                if candidate not in A:
                    new_A = A + [candidate]
                    new_counts = compute_counts_direct(new_A)
                    new_uncovered = int(np.sum(new_counts == 0))
                    old_uncovered = int(np.sum(counts == 0))
                    if new_uncovered < old_uncovered:
                        A = new_A
                        break
        return A

    def local_search(A, max_iters=500, use_smooth=False):
        """Improve A by trying single-element swaps and additions/removals."""
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
            for _ in range(15):
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
            if improved:
                continue
            if len(A) > 16:
                for idx in random.sample(range(len(A)), min(12, len(A))):
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
            if improved:
                continue
            if len(A) >= 2:
                for _ in range(15):
                    idx = random.randint(0, len(A) - 1)
                    candidate = random.randint(0, 128)
                    if candidate not in A:
                        new_A = A[:idx] + [candidate] + A[idx+1:]
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
        T = 15.0
        cooling = 0.9995
        min_T = 0.1

        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            if move_type < 0.3 and len(A) > 16:
                idx = random.randint(0, len(A) - 1)
                new_A = A[:idx] + A[idx+1:]
            elif move_type < 0.6:
                candidate = random.randint(0, 128)
                if candidate in A:
                    continue
                new_A = A + [candidate]
            else:
                if len(A) < 1:
                    continue
                idx = random.randint(0, len(A) - 1)
                candidate = random.randint(0, 128)
                if candidate in A:
                    continue
                new_A = A[:idx] + [candidate] + A[idx+1:]

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

    for k in range(16, 32):
        if time.time() > deadline - 15:
            break

        for restart in range(4):
            if time.time() > deadline - 10:
                break

            A = random_init(k)
            A = repair(A)
            if not is_valid(A):
                continue

            A = local_search(A, max_iters=150, use_smooth=True)
            obj = objective(A)

            sa_time = min(10, deadline - time.time() - 5)
            if sa_time > 2:
                A, sa_obj = simulated_annealing(A, time.time() + sa_time)
                obj = objective(A)

            if obj < best_score:
                best_score = obj
                best_A = A[:]

    if best_A is not None and time.time() < deadline - 3:
        best_A = local_search(best_A, max_iters=300, use_smooth=False)
        final_obj = objective(best_A)
        if final_obj < best_score:
            best_score = final_obj

    if best_A is None or not is_valid(best_A):
        return list(range(64, 129))

    return sorted(best_A)
