import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay well under 90s kill

    N = 256

    # 1. Precompute ALL APs (length >= 1) and reverse mapping
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

    # Precompute AP index lists as numpy arrays for fast fancy-indexing sum reads
    ap_arrays = [np.array(idx_list, dtype=np.intp) for idx_list in ap_indices]

    # Precompute affected AP arrays per position for fast flip updates
    affected_arrays = [np.array(aps_containing[i], dtype=np.intp) for i in range(N)]

    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            try:
                parity = i.bit_count() & 1
            except AttributeError:
                parity = bin(i).count('1') & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def compute_sums(signs):
        sums = np.empty(num_aps, dtype=np.int32)
        for r, arr in enumerate(ap_arrays):
            sums[r] = signs[arr].sum()
        return sums

    def greedy_optimize(s_init, time_budget, rng, no_improve_limit=5):
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))
        no_improve_sweeps = 0
        sweep_count = 0

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved_this_sweep = False
            sweep_count += 1

            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                aff = affected_arrays[i]

                old_sums = sums[aff].copy()

                # Apply flip
                sums[aff] += delta_val
                s[i] = -old_val

                # Fast path: did this flip change the global max?
                # Only affected APs changed. If none were at current_max and none
                # exceed it, the global max is unchanged.
                new_max = current_max
                if len(aff) > 0:
                    # If any affected AP was at the boundary, the max might drop
                    had_at_max = np.any(np.abs(old_sums) == current_max)
                    # If any affected AP now exceeds current_max, the max rises
                    now_exceeds = int(np.max(np.abs(sums[aff])))
                    if now_exceeds > current_max:
                        new_max = now_exceeds
                    elif had_at_max:
                        # Recompute global max (it may have dropped)
                        new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    improved_this_sweep = True
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                elif new_max == current_max:
                    # Tie: prefer flips that reduce the number of tight constraints
                    old_tight = int(np.sum(np.abs(old_sums) == current_max))
                    new_tight = int(np.sum(np.abs(sums[aff]) == current_max))
                    if new_tight < old_tight:
                        improved_this_sweep = True
                    elif new_tight == old_tight:
                        if rng.random() < 0.03:
                            improved_this_sweep = True
                        else:
                            sums[aff] = old_sums
                            s[i] = old_val
                    else:
                        sums[aff] = old_sums
                        s[i] = old_val
                else:
                    # new_max > current_max: revert
                    sums[aff] = old_sums
                    s[i] = old_val

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= no_improve_limit:
                    # Perturbation: flip a random subset to escape local minima
                    if rng.random() < 0.6:
                        num_flips = rng.randint(3, 12)
                        for _ in range(num_flips):
                            j = rng.choice(indices_list)
                            s[j] = -s[j]
                            sums[affected_arrays[j]] += -2 * s[j]
                        current_max = int(np.max(np.abs(sums)))
                        no_improve_sweeps = 0
                    else:
                        break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    # 3. Main loop: multiple restarts with different initializations
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Run 1: Thue-Morse (strong heuristic)
    tm = thue_morse(N)
    s1, score1 = greedy_optimize(tm, time_limit * 0.35, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Random init
    if time.time() - start_time < time_limit * 0.6:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.25, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Remaining runs: more random inits with shorter budgets
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 2.0:
        num_extra = min(8, int(remaining_budget / 2.5))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            budget = min(2.5, time_limit - (time.time() - start_time) - 1.0)
            if budget < 0.5:
                break
            s_k, score_k = greedy_optimize(rand_init, budget, rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
