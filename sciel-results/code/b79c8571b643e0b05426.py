"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 138.0
    N = 1000
    fft_size = 4096
    twoN = 2 * N

    # Precompute arrays used repeatedly
    all_sums = np.arange(2, twoN + 1)
    mids = all_sums // 2
    mid_valid = (mids >= 1) & (mids <= N)
    mid_in_set = np.zeros(twoN + 1, dtype=np.float64)

    def compute_counts_fft(indicator):
        """Compute rep counts via FFT convolution. Returns counts array indexed 0..2N-2 for sums 2..2N."""
        f = np.zeros(fft_size)
        f[1:N+1] = indicator[1:N+1].astype(float)
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        mid_in_set[1:N+1] = indicator[1:N+1].astype(float)
        diag = np.where(mid_valid, mid_in_set[mids], 0.0)
        reps = (conv[2:twoN+1] + diag) / 2.0
        return reps

    def add_element_check(counts, indicator, x):
        """Try adding x. Returns (valid, new_counts). counts indexed 0..2N-2 for sums 2..2N."""
        new_counts = counts.copy()
        y_values = np.where(indicator[1:N+1] > 0)[0] + 1
        for y in y_values:
            s = x + y
            if s <= twoN:
                new_counts[s - 2] += 1
                if new_counts[s - 2] > 2.0 + 1e-9:
                    return False, None
        s = 2 * x
        if s <= twoN:
            new_counts[s - 2] += 1
            if new_counts[s - 2] > 2.0 + 1e-9:
                return False, None
        return True, new_counts

    def remove_element_update(counts, indicator, x):
        """Remove x from set, update counts. Returns new_counts."""
        new_counts = counts.copy()
        y_values = np.where(indicator[1:N+1] > 0)[0] + 1
        for y in y_values:
            if y == x:
                continue
            s = x + y
            if s <= twoN:
                new_counts[s - 2] -= 1
        s = 2 * x
        if s <= twoN:
            new_counts[s - 2] -= 1
        indicator[x] = 0
        return new_counts

    # Greedy baseline: sequential
    chosen = []
    counts = np.zeros(twoN - 1, dtype=np.float64)
    indicator = np.zeros(N + 1, dtype=np.uint8)

    for x in range(1, N + 1):
        valid, new_counts = add_element_check(counts, indicator, x)
        if valid:
            chosen.append(x)
            counts = new_counts
            indicator[x] = 1

    best_set = list(chosen)
    best_size = len(chosen)
    best_counts = counts.copy()
    best_indicator = indicator.copy()

    np.random.seed(42)

    # Multiple greedy restarts with different orderings
    for attempt in range(80):
        if time.time() > deadline:
            break

        candidates = list(range(1, N + 1))
        np.random.shuffle(candidates)

        indicator = np.zeros(N + 1, dtype=np.uint8)
        counts = np.zeros(twoN - 1, dtype=np.float64)
        chosen = []

        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_counts = add_element_check(counts, indicator, x)
            if valid:
                chosen.append(x)
                counts = new_counts
                indicator[x] = 1

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
            best_counts = counts
            best_indicator = indicator

    # Local search: try to add elements, or swap to gain size
    if time.time() < deadline - 3:
        for _ in range(500):
            if time.time() > deadline:
                break
            x = np.random.randint(1, N + 1)
            if best_indicator[x] == 0:
                valid, new_counts = add_element_check(best_counts, best_indicator, x)
                if valid:
                    best_set.append(x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[x] = 1
            else:
                idx = np.random.randint(len(best_set))
                removed = best_set[idx]
                temp_indicator = best_indicator.copy()
                temp_counts = remove_element_update(best_counts.copy(), temp_indicator, removed)
                valid, new_counts = add_element_check(temp_counts, temp_indicator, x)
                if valid:
                    new_set = best_set[:idx] + best_set[idx+1:] + [x]
                    best_set = new_set
                    best_size = len(best_set)
                    best_counts = new_counts
                    best_indicator = temp_indicator
                    best_indicator[x] = 1

    # Final polish: try all remaining elements for addition
    if time.time() < deadline - 1:
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if best_indicator[x] == 0:
                valid, new_counts = add_element_check(best_counts, best_indicator, x)
                if valid:
                    best_set.append(x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[x] = 1

    return best_set
# EVOLVE-BLOCK-END
