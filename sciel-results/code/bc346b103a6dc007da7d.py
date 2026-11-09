# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(pt, order):
    """Compute makespan of a job permutation on a flow-shop.
    pt: (jobs, machines) array. order: 1-D array of job indices (permutation).
    Returns scalar makespan.
    """
    if len(order) == 0:
        return 0.0
    # completion time of each job on each machine, vectorized over machines
    # For a flow shop, completion C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # We can compute iteratively over jobs, vectorized over machines.
    jobs = order
    C = np.empty((len(jobs), pt.shape[1]), dtype=float)
    prev_job_end = 0.0
    for i, j in enumerate(jobs):
        p = pt[j]
        # machine m start = max(prev machine finish for this job, previous job finish on machine m)
        if i == 0:
            # first job: cumulative along machines
            C[i] = np.cumsum(p)
        else:
            prev = C[i - 1]
            # start_m = max(C[i-1][m-1] (this job's prev machine), prev[m] (prev job on this machine))
            # finish_m = start_m + p[m]
            # compute sequentially over machines but vectorized is tricky due to dependency.
            # Do a small loop over machines (machines usually small).
            nM = pt.shape[1]
            cur = np.empty(nM, dtype=float)
            for m in range(nM):
                if m == 0:
                    start = prev[0]
                else:
                    start = max(cur[m - 1], prev[m])
                cur[m] = start + p[m]
            C[i] = cur
    return float(C[-1, -1])


def _neh(pt, rng=None):
    """NEH heuristic: build permutation by inserting jobs in order of decreasing
    total processing time, choosing the best insertion position for makespan."""
    n = pt.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    totals = pt.sum(axis=1)
    order_of_jobs = np.argsort(-totals, kind='stable')
    # If rng provided, we can slightly perturb the initial ordering for diversity.
    if rng is not None:
        # small perturbation: with some probability swap adjacent in the sorted list
        # keep it mild
        for _ in range(n // 3):
            if n > 1:
                i = rng.integers(0, n - 1)
                order_of_jobs[i], order_of_jobs[i + 1] = order_of_jobs[i + 1], order_of_jobs[i]
    sched = [int(order_of_jobs[0])]
    for j in order_of_jobs[1:]:
        best_pos = 0
        best_mk = np.inf
        L = len(sched)
        for pos in range(L + 1):
            cand = sched[:pos] + [int(j)] + sched[pos:]
            mk = _makespan(pt, np.array(cand, dtype=int))
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        sched.insert(best_pos, int(j))
    return np.array(sched, dtype=int)


def _two_opt_eval(pt, order):
    """Return makespan for order (list of ints)."""
    return _makespan(pt, np.array(order, dtype=int))


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=float)
    n = pt.shape[0]
    if n <= 1:
        return list(range(n))
    deadline = time.time() + 100.0  # safety margin before 150s kill

    rng = np.random.default_rng(12345)

    # Initial solution via NEH
    best = _neh(pt)
    best_mk = _makespan(pt, best)

    # Iterated Local Search with 2-opt (swap) moves
    cur = best.copy()
    cur_mk = best_mk

    # Precompute nothing heavy; rely on fast makespan.
    # 2-opt: swap two positions.
    no_improve = 0
    max_no_improve = 200
    step = 0
    while time.time() < deadline - 5.0:
        # Local search: try all swaps, take best improving move (steepest descent)
        improved = False
        L = len(cur)
        best_delta = 0.0
        best_i, best_j = -1, -1
        # Evaluate swaps; for speed, only consider a subset if n large
        if n <= 400:
            pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
        else:
            # sample pairs
            pairs = [(int(rng.integers(0, L)), int(rng.integers(0, L))) for _ in range(20000)]
            # dedupe and ensure i<j
            seen = set()
            filt = []
            for i, j in pairs:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                filt.append((a, b))
            pairs = filt

        for i, j in pairs:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            # swap
            cur[a], cur[b] = cur[b], cur[a]
            mk = _makespan(pt, cur)
            delta = cur_mk - mk
            cur[a], cur[b] = cur[b], cur[a]  # revert
            if delta > best_delta:
                best_delta = delta
                best_i, best_j = a, b
        if best_delta > 1e-9 and best_i != -1:
            # apply best swap
            a, b = best_i, best_j
            cur[a], cur[b] = cur[b], cur[a]
            cur_mk -= best_delta
            improved = True
            if cur_mk < best_mk:
                best_mk = cur_mk
                best = cur.copy()
            no_improve = 0
        else:
            no_improve += 1
            # Perturb: random swap (or 2-opt reversal) to escape local optimum
            if no_improve >= max_no_improve:
                # stronger perturbation: random reversal of a segment
                L = len(cur)
                if L > 2:
                    i = int(rng.integers(0, L - 1))
                    j = int(rng.integers(i + 1, L))
                    cur[i:j + 1] = cur[i:j + 1][::-1]
                    cur_mk = _makespan(pt, cur)
                    no_improve = 0
                else:
                    i = int(rng.integers(0, L))
                    j = int(rng.integers(0, L))
                    cur[i], cur[j] = cur[j], cur[i]
                    cur_mk = _makespan(pt, cur)
                    no_improve = 0
            else:
                # mild perturbation: single random swap
                i = int(rng.integers(0, L))
                j = int(rng.integers(0, L))
                if i != j:
                    cur[i], cur[j] = cur[j], cur[i]
                    cur_mk = _makespan(pt, cur)
        step += 1

    return list(best)
# EVOLVE-BLOCK-END
