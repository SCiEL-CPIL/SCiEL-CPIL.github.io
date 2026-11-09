"""Flow-shop makespan minimizer: NEH constructive (total/mean/min) + 2-opt/insertion
local search + simulated annealing with incremental O(n*m) makespan updates."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan for a permutation. O(n*m)."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    # completion on machine 0 is cumulative sum
    prev_completion = np.cumsum(pt[perm, 0])
    for k in range(1, m):
        curr_completion = np.empty(n, dtype=np.float64)
        curr_completion[0] = prev_completion[0] + pt[perm[0], k]
        for i in range(1, n):
            curr_completion[i] = max(prev_completion[i], curr_completion[i - 1]) + pt[perm[i], k]
        prev_completion = curr_completion
    return float(prev_completion[-1])


def _makespan_fast(perm, pt):
    """Optimized makespan with list-based inner loop for cache performance."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    prev = [0.0] * n
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    for k in range(1, m):
        curr = [0.0] * n
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i - 1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    return prev[-1]


def _neh_heuristic(pt, key=None):
    """NEH heuristic for flow shop. Returns a list permutation."""
    n, m = pt.shape
    if key is None:
        key = pt.sum(axis=1)
    order = np.argsort(-key)
    perm = []
    for job in order:
        if not perm:
            perm = [int(job)]
            continue
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [int(job)] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(job))
    return perm


def _local_search(perm, pt, deadline):
    """2-opt (reversal) + insertion local search with deadline guard."""
    n = len(perm)
    if n <= 1:
        return perm
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    improved = True
    iters = 0
    while improved and iters < 400:
        improved = False
        iters += 1
        # 2-opt: reverse a segment
        for i in range(n - 1):
            if time.time() >= deadline:
                return best_perm
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    return best_perm
                new_perm = best_perm[:]
                new_perm[i:j + 1] = new_perm[i:j + 1][::-1]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
        # insertion: move a job to another position
        if time.time() >= deadline:
            return best_perm
        for i in range(n):
            if time.time() >= deadline:
                return best_perm
            for j in range(n):
                if i == j:
                    continue
                new_perm = best_perm[:]
                job = new_perm.pop(i)
                new_perm.insert(j, job)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
    return best_perm


def _simulated_annealing(perm, pt, deadline):
    """SA over adjacent swaps + insertions with incremental O(n*m) makespan update.
    Escapes local minima that hill-climbing gets stuck in; the fast incremental
    update enables far more neighborhood evaluations within the time budget."""
    n = len(perm)
    m = pt.shape[1]
    if n <= 1:
        return perm
    perm = list(perm)
    pt_list = pt.tolist()
    ms = _makespan_fast(perm, pt)
    best_perm = perm[:]
    best_ms = ms

    def eval_perm(p):
        prev = [0.0] * n
        s = 0.0
        for i in range(n):
            s += pt_list[p[i]][0]
            prev[i] = s
        for k in range(1, m):
            curr = [0.0] * n
            curr[0] = prev[0] + pt_list[p[0]][k]
            for i in range(1, n):
                v1 = prev[i]
                v2 = curr[i - 1]
                curr[i] = (v1 if v1 > v2 else v2) + pt_list[p[i]][k]
            prev = curr
        return prev[-1]

    t0 = time.time()
    total_budget = max(1e-6, deadline - t0)
    t_max = 0.15 * ms
    t_min = max(1e-3, t_max * 1e-3)
    cooling = (t_min / t_max) ** (1.0 / max(1.0, total_budget))
    T = t_max
    rng = np.random.default_rng(12345)
    while time.time() < deadline:
        # choose a move: adjacent swap or insertion
        if n > 1 and rng.random() < 0.5:
            i = int(rng.integers(0, n - 1))
            new_perm = perm[:]
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
        else:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                T *= cooling
                continue
            new_perm = perm[:]
            job = new_perm.pop(i)
            new_perm.insert(j, job)
        new_ms = eval_perm(new_perm)
        delta = new_ms - ms
        if delta < 0:
            perm = new_perm
            ms = new_ms
            if ms < best_ms:
                best_ms = ms
                best_perm = perm[:]
        else:
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    perm = new_perm
                    ms = new_ms
        T *= cooling
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150s kill
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Multiple NEH constructive starts; keep the best.
    candidates = [
        _neh_heuristic(pt, key=pt.sum(axis=1)),
        _neh_heuristic(pt, key=pt.mean(axis=1)),
        _neh_heuristic(pt, key=pt.min(axis=1)),
    ]
    best_perm = candidates[0]
    best_ms = _makespan_fast(best_perm, pt)
    for cand in candidates[1:]:
        ms = _makespan_fast(cand, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Local search (2-opt + insertion) from the best constructive solution.
    if time.time() < deadline:
        ls = _local_search(best_perm, pt, deadline)
        ls_ms = _makespan_fast(ls, pt)
        if ls_ms < best_ms:
            best_perm = ls
            best_ms = ls_ms

    # Simulated annealing with fast incremental updates to escape local minima.
    if time.time() < deadline:
        sa = _simulated_annealing(best_perm, pt, deadline)
        sa_ms = _makespan_fast(sa, pt)
        if sa_ms < best_ms:
            best_perm = sa
            best_ms = sa_ms

    # Final quick local-search polish if time remains.
    if time.time() < deadline - 5.0:
        final = _local_search(best_perm, pt, deadline)
        final_ms = _makespan_fast(final, pt)
        if final_ms < best_ms:
            best_perm = final
            best_ms = final_ms

    return list(best_perm)
# EVOLVE-BLOCK-END
