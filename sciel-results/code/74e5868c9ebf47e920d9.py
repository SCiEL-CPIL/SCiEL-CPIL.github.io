import time
import random
import numpy as np
import scipy.sparse as sp

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    N = 256

    # 1. Precompute APs and reverse mapping (index -> list of AP indices)
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(indices)

    num_aps = len(ap_indices)

    # Reverse mapping: for each position i in 0..N-1, which APs contain it?
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)

    # Convert to numpy int32 arrays for fast indexing
    aps_containing_np = [np.array(a, dtype=np.int32) for a in aps_containing]

    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            if count % 2 == 0:
                seq.append(1)
            else:
                seq.append(-1)
        return np.array(seq, dtype=np.int8)

    def greedy_optimize(s_init, time_budget, rng):
        """
        Perform Greedy Coordinate Descent with fast upper-bound checks.
        """
        s = s_init.copy()

        # Initialize sums
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])

        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

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

                # Upper-bound check: if no affected AP can reduce the global max,
                # skip the expensive full max computation.
                # After flip, new_sum = sums[affected] + delta_val
                # For improvement, we need max(|new_sum|) < current_max for ALL affected,
                # AND no unaffected AP exceeds current_max (they don't change).
                # Quick reject: if any affected AP currently has |sum| >= current_max
                # and flipping doesn't reduce its |sum|, we can't improve.
                # More precisely: if the max |sum| among affected is already < current_max
                # and flipping keeps them < current_max, then the global max is unchanged
                # (since unaffected don't change). So no improvement possible.
                # Actually, improvement requires the NEW global max < current_max.
                # The new global max = max(max_unaffected, max_affected_new)
                # max_unaffected = current_max if there's an unaffected AP at current_max,
                #   else the max of unaffected.
                # This is complex. Simpler heuristic:
                # If max(|sums[affected]|) < current_max, then flipping can only reduce
                # those further or keep them same magnitude. The global max stays at
                # current_max (from unaffected APs). No improvement. Skip.
                # If max(|sums[affected]|) == current_max, flipping MIGHT reduce it.
                # If max(|sums[affected]|) > current_max... that can't happen since
                #   current_max is the global max.
                # So: if max(|sums[affected]|) < current_max, skip (no improvement possible).
                # This is a valid quick-reject.

                affected_sums = sums[affected]
                if np.max(np.abs(affected_sums)) < current_max:
                    # Flipping this index cannot reduce the global max
                    # (it only affects APs that are already below the max)
                    continue

                # Full evaluation: apply flip, compute new max, revert if not better
                old_sums = affected_sums.copy()
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                elif new_max == current_max:
                    # Accept with small probability to allow exploration
                    if rng.random() < 0.05:
                        improved = True
                    else:
                        sums[affected] = old_sums
                        s[i] = old_val
                else:
                    sums[affected] = old_sums
                    s[i] = old_val

            if not improved:
                break

        return best_s, best_score

    def two_opt_local_search(s_init, time_budget, rng):
        """
        Try swapping pairs of signs to escape local minima.
        Only attempt swaps where both positions have affected APs at current_max.
        """
        s = s_init.copy()

        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])

        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        while time.time() - start_time < time_budget:
            improved = False
            # Try a random set of swaps
            for _ in range(200):
                if time.time() - start_time > time_budget:
                    break

                i, j = rng.sample(range(N), 2)
                if s[i] == s[j]:
                    continue

                # Swapping i and j: if s[i]=1, s[j]=-1, then s[i] becomes -1, s[j] becomes 1
                # This is equivalent to flipping both i and j.
                # Delta for APs containing i but not j: -2*s[i]
                # Delta for APs containing j but not i: -2*s[j]
                # Delta for APs containing both: 0 (net change is 0)

                affected_i = aps_containing_np[i]
                affected_j = aps_containing_np[j]

                # Quick reject: if neither position affects any AP at current_max, skip
                max_i = int(np.max(np.abs(sums[affected_i])))
                max_j = int(np.max(np.abs(sums[affected_j])))
                if max_i < current_max and max_j < current_max:
                    continue

                # Apply swap
                old_si = s[i]
                old_sj = s[j]
                s[i] = -old_si
                s[j] = -old_sj

                # Update sums
                # APs containing i: change by -2*old_si
                sums[affected_i] += -2 * old_si
                # APs containing j: change by -2*old_sj
                sums[affected_j] += -2 * old_sj
                # APs containing both were updated twice (once for i, once for j),
                # which is correct since they contain both positions.

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                else:
                    # Revert
                    sums[affected_i] += 2 * old_si
                    sums[affected_j] += 2 * old_sj
                    s[i] = old_si
                    s[j] = old_sj

            if not improved:
                break

        return best_s, best_score

    # Main Loop: Multiple Restarts
    best_global_s = None
    best_global_score = float('inf')

    rng_global = random.Random(42)

    restarts = 0
    while time.time() - start_time < time_limit:
        if restarts == 0:
            s_init = thue_morse(N)
        else:
            s_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break

        # Allocate time: 70% to greedy, 30% to 2-opt
        greedy_budget = remaining * 0.7
        two_opt_budget = remaining * 0.3

        s_best_run, score_best_run = greedy_optimize(s_init, greedy_budget, rng_global)

        # If we have time, try 2-opt on the best result
        if time.time() - start_time < time_limit - 0.5:
            s_best_run, score_best_run = two_opt_local_search(
                s_best_run, time_limit - 0.5, rng_global
            )

        if score_best_run < best_global_score:
            best_global_score = score_best_run
            best_global_s = s_best_run

        restarts += 1

        if best_global_score <= 1:
            break

    if best_global_s is None:
        best_global_s = thue_morse(N)

    return [int(x) for x in best_global_s]
