"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(perm, pt):
    """Fast makespan: perm is a list of job indices, pt is [n, m] array."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        row = pt[job_idx]
        prev_c = 0.0
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]

def _neh(processing_times):
    """NEH heuristic: insert jobs by decreasing total processing time."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals).tolist()

    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    return perm

def _insertion_local_search(perm, pt, max_iters=200, time_budget=None, start_time=None, safety=1.0):
    """Insertion-based local search (move job to any other position).
    First-improvement per pass; stops when a full pass yields no improvement
    or when the time budget (with safety margin) is exhausted."""
    n = len(perm)
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)

    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current_perm[i]
            # remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            for k in range(n - 1):
                if k == i:
                    continue
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
                if ms < current_ms:
                    current_perm = trial
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
        if time_budget is not None and start_time is not None:
            if time.time() > time_budget - safety:
                break
    return current_perm, current_ms

def _tabu_search(perm, pt, tabu_len, time_budget, start_time, safety=1.0, max_iters=4000):
    """Tabu search over insertion moves with a recency tabu list on the
    (job, target-position) pair, plus aspiration (accept tabu move if it beats
    the incumbent). Escapes local optima that greedy insertion cannot."""
    n = len(perm)
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    best_perm = current_perm[:]
    best_ms = current_ms

    # tabu: dict (job, target_pos) -> iteration when it becomes non-tabu
    tabu = {}
    rng = random.Random(1234)

    for it in range(max_iters):
        if time.time() > time_budget - safety:
            break

        best_trial = None
        best_trial_ms = float('inf')
        best_trial_key = None
        best_trial_tabu_until = 0

        for i in range(n):
            job = current_perm[i]
            without_job = current_perm[:i] + current_perm[i+1:]
            for k in range(n - 1):
                if k == i:
                    continue
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
                key = (job, k)
                tabu_until = tabu.get(key, 0)
                # accept if improving, or if best-so-far, or if aspiration (beats incumbent)
                is_tabu = it < tabu_until
                if not is_tabu or ms < best_ms:
                    if ms < best_trial_ms:
                        best_trial_ms = ms
                        best_trial = trial
                        best_trial_key = key
                        best_trial_tabu_until = tabu_until

        if best_trial is None:
            # all moves tabu and none aspiration: diversify with a random swap
            i, j = rng.sample(range(n), 2)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_ms = _makespan_fast(current_perm, pt)
            continue

        # set tabu for the chosen move
        tabu[best_trial_key] = it + tabu_len
        current_perm = best_trial
        current_ms = best_trial_ms

        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm[:]

    return best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin within 150s wall-clock

    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))

    pt = processing_times

    # 1. Initial solution: NEH
    best_perm = _neh(pt)
    best_ms = _makespan_fast(best_perm, pt)

    # 2. Polish with insertion local search
    best_perm, best_ms = _insertion_local_search(
        best_perm, pt, max_iters=300, time_budget=deadline, start_time=start_time)

    # 3. Tabu search to escape the initial local optimum
    best_perm, best_ms = _tabu_search(
        best_perm, pt, tabu_len=max(5, n // 3), time_budget=deadline, start_time=start_time)

    # 4. Multi-restart Simulated Annealing + insertion polish, adaptive to time
    rng = np.random.RandomState(42)

    # estimate cost of one SA step (makespan eval ~ O(n*m))
    t0 = time.time()
    probe_steps = 20
    for _ in range(probe_steps):
        _makespan_fast(best_perm, pt)
    step_cost = max(1e-6, (time.time() - t0) / probe_steps)

    # Decide how many restarts we can afford, leaving room for polish + tabu
    # Each restart: ~2000 SA steps + one insertion LS pass (bounded).
    steps_per_restart = 2000
    # rough per-restart cost
    per_restart_cost = steps_per_restart * step_cost + 2.0  # + overhead for LS
    remaining = deadline - time.time() - 5.0
    n_restarts = max(1, int(remaining // per_restart_cost))
    n_restarts = min(n_restarts, 25)  # cap to avoid over-spending on tiny instances

    for restart in range(n_restarts):
        if time.time() > deadline - 6.0:
            break

        # Perturb the best solution for a diverse starting point
        current_perm = list(best_perm)
        num_swaps = max(1, n // 5)
        for _ in range(num_swaps):
            i, j = rng.choice(n, 2, replace=False)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        current_ms = _makespan_fast(current_perm, pt)

        # Simulated Annealing
        T = best_ms * 0.05
        T_min = best_ms * 0.001
        alpha = 0.9998

        for _ in range(steps_per_restart):
            if time.time() > deadline - 4.0:
                break
            if T < T_min:
                break

            i, j = rng.choice(n, 2, replace=False)
            new_perm = current_perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

            new_ms = _makespan_fast(new_perm, pt)
            delta = new_ms - current_ms

            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
            T *= alpha

        # Polish the SA result if it's reasonably close to the incumbent
        if time.time() > deadline - 8.0:
            break
        if current_ms <= best_ms * 1.15:
            polished_perm, polished_ms = _insertion_local_search(
                current_perm, pt, max_iters=150, time_budget=deadline, start_time=start_time)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm

    # 5. Final tabu polish on the incumbent to squeeze out remaining gains
    if time.time() < deadline - 5.0:
        final_perm, final_ms = _tabu_search(
            best_perm, pt, tabu_len=max(5, n // 3),
            time_budget=deadline, start_time=start_time, safety=3.0, max_iters=2000)
        if final_ms < best_ms:
            best_ms = final_ms
            best_perm = final_perm

    return list(best_perm)
# EVOLVE-BLOCK-END
