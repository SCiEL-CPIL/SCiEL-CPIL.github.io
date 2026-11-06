"""Rewritten flow-shop heuristic: NEH + fast 2-opt and insertion local search, plus SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a permutation of job indices."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0

    prev = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            # vectorize a little: cur[m] = max(prev[m], cur[m-1]) + pt[m]
            # still sequential in m, but with numpy scalars it's okay
            for m in range(1, n_machines):
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
        prev = cur
    return float(prev[-1])


def _prefix_completions(perm, processing_times):
    """Return list of n_jobs+1 completion vectors for prefixes of perm."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix = [None] * (n_jobs + 1)
    prefix[0] = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        prev = prefix[j]
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
        prefix[j + 1] = cur
    return prefix


def _swap_makespan(perm, i, prefix, processing_times):
    """Makespan after swapping perm[i] and perm[i+1] using prefix[i]."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C_prev = prefix[i]
    pt_b = processing_times[perm[i + 1]]
    pt_a = processing_times[perm[i]]

    # job b first
    C_b = np.empty(n_machines, dtype=np.float64)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = (C_prev[m] if C_prev[m] > C_b[m - 1] else C_b[m - 1]) + pt_b[m]

    # job a second
    C_a = np.empty(n_machines, dtype=np.float64)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = (C_b[m] if C_b[m] > C_a[m - 1] else C_a[m - 1]) + pt_a[m]

    # continue with remaining jobs
    cur = C_a
    for j in range(i + 2, n_jobs):
        pt = processing_times[perm[j]]
        nxt = np.empty(n_machines, dtype=np.float64)
        nxt[0] = cur[0] + pt[0]
        for m in range(1, n_machines):
            nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
        cur = nxt
    return float(cur[-1])


def _insert_makespan(perm, i, j, processing_times):
    """Makespan of moving perm[i] to position j in the list after removal."""
    new_perm = list(perm)
    job = new_perm.pop(i)
    new_perm.insert(j, job)
    return _makespan(new_perm, processing_times)


def _neh(processing_times, rng):
    """NEH constructive heuristic."""
    n_jobs, _ = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))

    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))

    partial = [order[0]]
    for k in range(1, n_jobs):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        # try each insertion position
        for pos in range(len(partial) + 1):
            cand = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(cand, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    return partial


def _local_search(perm, processing_times, deadline, rng, n_jobs_cap_insert=200):
    """2-opt + insertion local search with incremental swap eval."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        prefix = _prefix_completions(best_perm, processing_times)
        # 2-opt: scan all adjacent swaps, take best
        best_i = -1
        best_swap_ms = best_ms
        for i in range(n_jobs - 1):
            ms = _swap_makespan(best_perm, i, prefix, processing_times)
            if ms < best_swap_ms - 1e-12:
                best_swap_ms = ms
                best_i = i
        if best_i >= 0:
            best_perm[best_i], best_perm[best_i + 1] = best_perm[best_i + 1], best_perm[best_i]
            best_ms = best_swap_ms
            improved = True
            continue

        # insertion moves
        if n_jobs <= n_jobs_cap_insert:
            best_move = None
            best_ins_ms = best_ms
            # randomize start order to avoid deterministic cycling
            idx_order = list(range(n_jobs))
            rng.shuffle(idx_order)
            for i in idx_order:
                if time.time() > deadline - 1.0:
                    break
                job = best_perm[i]
                # positions where inserting job is not same as current
                for j in range(n_jobs):
                    if j == i or j == i + 1:
                        continue
                    ms = _insert_makespan(best_perm, i, j, processing_times)
                    if ms < best_ins_ms - 1e-12:
                        best_ins_ms = ms
                        best_move = (i, j, ms)
                if best_move is not None:
                    break
            if best_move is not None:
                i, j, ms = best_move
                job = best_perm.pop(i)
                best_perm.insert(j, job)
                best_ms = ms
                improved = True

    return best_perm, best_ms


def _simulated_annealing(perm, ms, processing_times, deadline, rng):
    """SA with 2-opt and insertion moves."""
    n_jobs = len(perm)
    cur_perm = list(perm)
    cur_ms = ms
    best_perm = list(cur_perm)
    best_ms = cur_ms

    T = cur_ms * 0.05
    T_min = max(cur_ms * 0.001, 1e-6)
    cool = 0.998

    while time.time() < deadline and T > T_min:
        prefix = _prefix_completions(cur_perm, processing_times)
        if rng.random() < 0.6:
            i = rng.randint(0, n_jobs - 2)
            new_ms = _swap_makespan(cur_perm, i, prefix, processing_times)
            new_perm = list(cur_perm)
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
        else:
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                T *= cool
                continue
            new_perm = list(cur_perm)
            job = new_perm.pop(i)
            new_perm.insert(j, job)
            new_ms = _makespan(new_perm, processing_times)

        delta = new_ms - cur_ms
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur_perm = new_perm
            cur_ms = new_ms
            if cur_ms < best_ms:
                best_ms = cur_ms
                best_perm = list(cur_perm)
        T *= cool

    return best_perm, best_ms


def order_jobs(processing_times):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s

    n_jobs, _ = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    rng = random.Random(123)

    # generate a few diverse starting solutions
    candidates = []
    candidates.append(_neh(processing_times, rng))
    # random NEH with perturbed order
    for _ in range(2):
        rng2 = random.Random(rng.randint(0, 10**9))
        candidates.append(_neh(processing_times, rng2))
    # LPT
    candidates.append(list(np.argsort(-processing_times.sum(axis=1))))

    best_perm = candidates[0]
    best_ms = _makespan(best_perm, processing_times)
    for p in candidates[1:]:
        ms = _makespan(p, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = p

    # local search phase
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search(best_perm, processing_times, deadline - 5.0, rng)

    # simulated annealing if time remains
    if time.time() < deadline - 2.0:
        best_perm, best_ms = _simulated_annealing(best_perm, best_ms, processing_times, deadline - 2.0, rng)

    # final polish
    if time.time() < deadline - 0.5:
        best_perm, best_ms = _local_search(best_perm, processing_times, deadline - 0.5, rng, n_jobs_cap_insert=0)

    return list(best_perm)
# EVOLVE-BLOCK-END
