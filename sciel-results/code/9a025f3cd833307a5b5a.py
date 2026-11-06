import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    N = 256

    # Precompute all arithmetic progressions
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(indices)

    num_aps = len(ap_indices)

    # Reverse mapping: for each position i, which APs contain it?
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)

    aps_containing_np = [np.array(a, dtype=np.int32) for a in aps_containing]

    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            seq.append(1 if count % 2 == 0 else -1)
        return np.array(seq, dtype=np.int8)

    def compute_sums(s):
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
        return sums

    def greedy_cd(s_init, time_budget, rng):
        """
        Greedy Coordinate Descent with efficient tracking.
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        # Track which APs are at the current max
        at_max = np.abs(sums) >= current_max

        indices_list = list(range(N))

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved = False

            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = aps_containing_np[i]

                # Quick reject: if no affected AP is at current max, skip
                if not np.any(at_max[affected]):
                    continue

                # Apply flip
                old_sums_affected = sums[affected].copy()
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    at_max = np.abs(sums) >= current_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                elif new_max == current_max:
                    # Accept with small probability for exploration
                    if rng.random() < 0.03:
                        at_max = np.abs(sums) >= current_max
                        improved = True
                    else:
                        sums[affected] = old_sums_affected
                        s[i] = old_val
                else:
                    sums[affected] = old_sums_affected
                    s[i] = old_val

            if not improved:
                # Try a random perturbation to escape local minimum
                if rng.random() < 0.1:
                    for _ in range(rng.randint(1, 3)):
                        j = rng.randint(0, N - 1)
                        old_val = s[j]
                        delta_val = -2 * old_val
                        affected = aps_containing_np[j]
                        sums[affected] += delta_val
                        s[j] = -old_val

        return best_s, best_score

    def pair_swap_search(s_init, time_budget, rng):
        """
        Local search using pair swaps to escape local minima.
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        while time.time() - start_time < time_budget:
            improved = False
            attempts = 0

            while attempts < 500 and time.time() - start_time < time_budget:
                attempts += 1
                i, j = rng.sample(range(N), 2)
                if s[i] == s[j]:
                    continue

                affected_i = aps_containing_np[i]
                affected_j = aps_containing_np[j]

                # Quick reject
                max_i = int(np.max(np.abs(sums[affected_i])))
                max_j = int(np.max(np.abs(sums[affected_j])))
                if max_i < current_max and max_j < current_max:
                    continue

                # Apply swap (flip both)
                old_si = s[i]
                old_sj = s[j]
                old_sums_i = sums[affected_i].copy()
                old_sums_j = sums[affected_j].copy()

                sums[affected_i] += -2 * old_si
                sums[affected_j] += -2 * old_sj
                s[i] = -old_si
                s[j] = -old_sj

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                else:
                    sums[affected_i] = old_sums_i
                    sums[affected_j] = old_sums_j
                    s[i] = old_si
                    s[j] = old_sj

            if not improved:
                break

        return best_s, best_score

    # Main: Multiple restarts with different strategies
    best_global_s = None
    best_global_score = float('inf')

    rng_global = random.Random(42)

    # Start with Thue-Morse
    s_init = thue_morse(N)
    s_best, score_best = greedy_cd(s_init, 20.0, rng_global)
    best_global_s = s_best
    best_global_score = score_best

    # Continue with pair swaps
    if time.time() - start_time < time_limit - 1.0:
        s_best, score_best = pair_swap_search(s_best, time_limit - 1.0, rng_global)
        if score_best < best_global_score:
            best_global_score = score_best
            best_global_s = s_best

    # Additional restarts with random initializations
    restarts = 0
    while time.time() - start_time < time_limit - 2.0:
        s_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        remaining = time_limit - 2.0 - (time.time() - start_time)
        if remaining < 2.0:
            break

        # Allocate time
        greedy_time = remaining * 0.6
        swap_time = remaining * 0.4

        s_best, score_best = greedy_cd(s_init, greedy_time, rng_global)
        if time.time() - start_time < time_limit - 1.0:
            s_best, score_best = pair_swap_search(s_best, time_limit - 1.0, rng_global)

        if score_best < best_global_score:
            best_global_score = score_best
            best_global_s = s_best

        restarts += 1

        if best_global_score <= 1:
            break

    if best_global_s is None:
        best_global_s = thue_morse(N)

    return [int(x) for x in best_global_s]
