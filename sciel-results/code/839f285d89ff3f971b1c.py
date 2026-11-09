"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_fast(times, pred_masks, cycle_time, priorities):
    n = len(times)
    assigned_mask = 0
    remaining_mask = (1 << n) - 1
    stations = 0

    while remaining_mask:
        stations += 1
        current_time = 0.0

        while True:
            best_task = -1
            best_prio = -1.0

            for i in range(n):
                if not (remaining_mask & (1 << i)):
                    continue
                if (assigned_mask & pred_masks[i]) == pred_masks[i]:
                    if times[i] <= cycle_time - current_time:
                        if priorities[i] > best_prio:
                            best_prio = priorities[i]
                            best_task = i

            if best_task == -1:
                break

            assigned_mask |= (1 << best_task)
            remaining_mask &= ~(1 << best_task)
            current_time += times[best_task]

    return stations


def _simulate_fast_rank(times, pred_masks, cycle_time, rank, succ_count):
    """
    Greedy simulation that breaks ties by rank (longest path to end),
    then by successor count, then by task index.
    This avoids needing a full priority array for the base heuristic.
    """
    n = len(times)
    assigned_mask = 0
    remaining_mask = (1 << n) - 1
    stations = 0

    while remaining_mask:
        stations += 1
        current_time = 0.0

        while True:
            best_task = -1
            best_rank = -1.0
            best_succ = -1
            best_idx = -1

            for i in range(n):
                if not (remaining_mask & (1 << i)):
                    continue
                if (assigned_mask & pred_masks[i]) == pred_masks[i]:
                    if times[i] <= cycle_time - current_time:
                        r = rank[i]
                        s = succ_count[i]
                        # Higher rank is better, then higher succ count, then lower index
                        if (r > best_rank or
                            (r == best_rank and s > best_succ) or
                            (r == best_rank and s == best_succ and (best_idx == -1 or i < best_idx))):
                            best_rank = r
                            best_succ = s
                            best_idx = i
                            best_task = i

            if best_task == -1:
                break

            assigned_mask |= (1 << best_task)
            remaining_mask &= ~(1 << best_task)
            current_time += times[best_task]

    return stations


def _compute_rank(times, succ, n):
    """Compute longest path from each node to a sink (rank)."""
    memo = [-1.0] * n
    in_stack = [False] * n

    def get_rank(i):
        if memo[i] != -1.0:
            return memo[i]
        in_stack[i] = True
        if not succ[i]:
            memo[i] = times[i]
            in_stack[i] = False
            return memo[i]
        max_succ = 0.0
        for s in succ[i]:
            val = get_rank(s)
            if val > max_succ:
                max_succ = val
        memo[i] = times[i] + max_succ
        in_stack[i] = False
        return memo[i]

    for i in range(n):
        if memo[i] == -1.0:
            get_rank(i)

    return memo


def _compute_succ_count(succ, n):
    """Count number of direct successors for each task."""
    return [len(succ[i]) for i in range(n)]


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Precompute predecessor bitmasks
    pred_masks = [0] * n
    for i in range(n):
        mask = 0
        for p in preds[i]:
            mask |= (1 << p)
        pred_masks[i] = mask

    # Build successors
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)

    # Compute rank and successor count
    rank = _compute_rank(times, succ, n)
    succ_count = _compute_succ_count(succ, n)

    # Initial priorities based on rank with small tie-breakers
    base_priorities = np.array([rank[i] + i * 1e-9 for i in range(n)], dtype=float)

    # Evaluate base heuristic
    best_stations = _simulate_fast_rank(times, pred_masks, cycle_time, rank, succ_count)
    best_priorities = base_priorities.copy()

    # Local search: swap-based perturbation on priority vectors
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s kill

    rng = np.random.default_rng(42)

    # Track consecutive non-improvements for adaptive cooling
    no_improve_count = 0
    max_no_improve = 200

    while time.time() - start_time < time_limit:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue

        # Swap priorities
        old_i = best_priorities[i]
        old_j = best_priorities[j]
        best_priorities[i] = old_j
        best_priorities[j] = old_i

        new_stations = _simulate_fast(times, pred_masks, cycle_time, best_priorities)

        if new_stations < best_stations:
            best_stations = new_stations
            no_improve_count = 0
        else:
            # Revert
            best_priorities[i] = old_i
            best_priorities[j] = old_j
            no_improve_count += 1

        # If stuck for too long, try a larger perturbation (3-swap)
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Try a random 3-element shuffle
            idx = rng.choice(n, size=min(3, n), replace=False)
            old_vals = [best_priorities[k] for k in idx]
            perm = rng.permutation(len(idx))
            for k, p in enumerate(perm):
                best_priorities[idx[k]] = old_vals[p]

            new_stations = _simulate_fast(times, pred_masks, cycle_time, best_priorities)

            if new_stations < best_stations:
                best_stations = new_stations
            else:
                # Revert
                for k in range(len(idx)):
                    best_priorities[idx[k]] = old_vals[k]

    return best_priorities
# EVOLVE-BLOCK-END
