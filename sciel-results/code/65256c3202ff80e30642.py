import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay well under 90s kill

    N = 256

    # 1. Precompute ALL APs (length >= 1) and build incidence structure
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d

            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)

    num_aps = len(ap_indices)
    # Max possible |sum| is N (all same sign in a length-N AP)
    max_abs_sum = N  # 256

    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def compute_sums_and_counts(signs):
        """Compute sums for all APs and build count array."""
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(signs[idx_list])
        counts = np.bincount(np.abs(sums), minlength=max_abs_sum + 1)
        # Find current max
        current_max = max_abs_sum
        while current_max > 0 and counts[current_max] == 0:
            current_max -= 1
        return sums, counts, current_max

    def greedy_optimize(s_init, time_budget, rng, strict_polish=False):
        """
        Greedy Coordinate Descent with O(1) max tracking via count array.
        strict_polish: if True, only accept strict improvements (no tie-breaking).
        """
        s = s_init.copy()
        sums, counts, current_max = compute_sums_and_counts(s)
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))
        no_improve_sweeps = 0
        max_no_improve = 5 if not strict_polish else 3

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved_this_sweep = False
            sweep_start = time.time()

            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = aps_containing[i]

                # Save old sums for affected APs
                old_sums = sums[affected].copy()

                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val

                # Update counts: for each affected AP, decrement old |sum|, increment new |sum|
                # Check if max decreased
                new_max = current_max
                # Only need to check if the max level lost all its members
                if counts[current_max] > 0:
                    # Check affected APs that were at current_max
                    at_max_old = old_sums == current_max
                    at_max_old_neg = old_sums == -current_max
                    at_max_new = sums[affected] == current_max
                    at_max_new_neg = sums[affected] == -current_max

                    # Net change in count at current_max
                    removed = np.sum(at_max_old | at_max_old_neg)
                    added = np.sum(at_max_new | at_max_new_neg)
                    if removed > added:
                        # Max might decrease
                        counts[current_max] -= (removed - added)
                        if counts[current_max] == 0:
                            # Find new max
                            while new_max > 0 and counts[new_max] == 0:
                                new_max -= 1
                            current_max = new_max
                            improved_this_sweep = True
                            if current_max < best_score:
                                best_score = current_max
                                best_s = s.copy()
                    else:
                        counts[current_max] += (added - removed)
                else:
                    # current_max level was empty, shouldn't happen if we maintain correctly
                    # But just in case, recompute
                    pass

                # Update counts for all affected APs (full update)
                for k in range(len(affected)):
                    old_abs = abs(old_sums[k])
                    new_abs = abs(sums[affected][k])
                    if old_abs != new_abs:
                        counts[old_abs] -= 1
                        counts[new_abs] += 1

                # Re-check max after full count update
                # If we didn't improve above, check if max could have increased (it shouldn't since we only flip)
                # Actually flipping can increase some sums, so max could increase
                # Check if any affected AP now exceeds current_max
                max_affected_new = int(np.max(np.abs(sums[affected]))) if len(affected) > 0 else 0
                if max_affected_new > current_max:
                    current_max = max_affected_new
                    # Revert: this flip made things worse
                    sums[affected] = old_sums
                    s[i] = old_val
                    # Restore counts
                    for k in range(len(affected)):
                        old_abs = abs(old_sums[k])
                        new_abs = abs(sums[affected][k])
                        if old_abs != new_abs:
                            counts[old_abs] -= 1
                            counts[new_abs] += 1
                elif not strict_polish and current_max == best_score:
                    # Tie-breaking: accept if it reduces tightness
                    old_tight = int(np.sum(np.abs(old_sums) == current_max))
                    new_tight = int(np.sum(np.abs(sums[affected]) == current_max))
                    if new_tight < old_tight:
                        improved_this_sweep = True
                    elif new_tight == old_tight and rng.random() < 0.03:
                        improved_this_sweep = True
                    else:
                        # Revert
                        sums[affected] = old_sums
                        s[i] = old_val
                        for k in range(len(affected)):
                            old_abs = abs(old_sums[k])
                            new_abs = abs(sums[affected][k])
                            if old_abs != new_abs:
                                counts[old_abs] -= 1
                                counts[new_abs] += 1

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    if not strict_polish and rng.random() < 0.6:
                        num_flips = rng.randint(4, 12)
                        for _ in range(num_flips):
                            j = rng.choice(indices_list)
                            s[j] = -s[j]
                            delta = -2 * s[j]
                            affected_j = aps_containing[j]
                            old_sums_j = sums[affected_j].copy()
                            sums[affected_j] += delta
                            for k in range(len(affected_j)):
                                old_abs = abs(old_sums_j[k])
                                new_abs = abs(sums[affected_j][k])
                                if old_abs != new_abs:
                                    counts[old_abs] -= 1
                                    counts[new_abs] += 1
                        current_max = int(np.max(np.abs(sums)))
                        no_improve_sweeps = 0
                    else:
                        break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    # Main loop
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Run 1: Thue-Morse
    tm = thue_morse(N)
    s1, score1 = greedy_optimize(tm, time_limit * 0.35, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Random init
    if time.time() - start_time < time_limit * 0.55:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.2, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # More random inits
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 2.0:
        num_extra = min(10, int(remaining_budget / 2.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 3.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            budget = min(2.0, time_limit - (time.time() - start_time) - 3.0)
            if budget < 0.5:
                break
            s_k, score_k = greedy_optimize(rand_init, budget, rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final polish pass: strict improvement only, to squeeze out last units
    if best_global_s is not None and time.time() - start_time < time_limit - 2.0:
        polish_budget = time_limit - (time.time() - start_time) - 1.0
        if polish_budget > 1.0:
            s_polish, score_polish = greedy_optimize(best_global_s.copy(), polish_budget, rng_global, strict_polish=True)
            if score_polish < best_global_score:
                best_global_score = score_polish
                best_global_s = s_polish.copy()

    # Final safety
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
