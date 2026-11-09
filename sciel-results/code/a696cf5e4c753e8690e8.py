import numpy as np
import time
import random

def construct():
    deadline = time.time() + 85
    n = 128
    target_lo, target_hi = 128, 256
    n_targets = target_hi - target_lo + 1

    def compute_peak_fast(A_arr):
        k = len(A_arr)
        if k == 0:
            return 10**9
        sums = A_arr[:, None] + A_arr[None, :]
        iu = np.triu_indices(k)
        pair_sums = sums[iu]
        mask = (pair_sums >= target_lo) & (pair_sums <= target_hi)
        in_range_sums = pair_sums[mask]
        if len(in_range_sums) < n_targets:
            return 10**9
        counts = np.bincount(in_range_sums - target_lo, minlength=n_targets)
        if counts.min() == 0:
            return 10**9
        return int(counts.max())

    best_A = None
    best_peak = 10**9

    # Seed
    A0 = np.array(list(range(64, 129)), dtype=np.int64)
    p = compute_peak_fast(A0)
    if p < best_peak:
        best_peak = p
        best_A = A0.copy()

    # Random starts
    np.random.seed(42)
    for trial in range(100):
        size = np.random.randint(25, 75)
        A = np.sort(np.random.choice(129, size, replace=False)).astype(np.int64)
        p = compute_peak_fast(A)
        if p < best_peak:
            best_peak = p
            best_A = A.copy()

    # Local search with simulated annealing
    random.seed(42)
    current_A = best_A.copy()
    current_peak = best_peak
    T = 5.0

    while time.time() < deadline:
        op = random.random()
        A_set = set(current_A.tolist())

        if op < 0.3 and len(A_set) > 15:
            x = random.choice(list(A_set))
            A_set.remove(x)
        elif op < 0.6:
            avail = [x for x in range(129) if x not in A_set]
            if not avail:
                continue
            A_set.add(random.choice(avail))
        else:
            avail = [x for x in range(129) if x not in A_set]
            if not avail or len(A_set) == 0:
                continue
            to_remove = random.choice(list(A_set))
            A_set.remove(to_remove)
            A_set.add(random.choice(avail))

        new_A = np.array(sorted(A_set), dtype=np.int64)
        new_peak = compute_peak_fast(new_A)

        if new_peak <= current_peak or (new_peak < 10**9 and random.random() < np.exp((current_peak - new_peak) / max(T, 0.01))):
            current_A = new_A
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = new_A.copy()

        T *= 0.9995

    return list(best_A)
