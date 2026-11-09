# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(pt, perm):
    """Compute makespan for a given job permutation.

    pt: (jobs, machines) processing times.
    perm: (n,) array of job indices in sequence.
    Returns scalar makespan.
    """
    n = len(perm)
    if n == 0:
        return 0.0
    # completion times matrix: machines x jobs-in-sequence
    # C[k][j] = completion of job j (in sequence position) on machine k
    # C[0][j] = C[0][j-1] + pt[perm[j], 0]
    # C[k][j] = max(C[k-1][j], C[k][j-1]) + pt[perm[j], k]
    m = pt.shape[1]
    # work with the submatrix of selected jobs in order
    sub = pt[perm]  # (n, m)
    # C has shape (m, n); C[k, j]
    C = np.empty((m, n), dtype=np.float64)
    first_col = sub[:, 0]
    C[0, 0] = first_col[0]
    for j in range(1, n):
        C[0, j] = C[0, j - 1] + sub[j, 0]
    for k in range(1, m):
        C[k, 0] = C[k - 1, 0] + sub[k, 0]
        for j in range(1, n):
            C[k, j] = max(C[k - 1, j], C[k, j - 1]) + sub[k, j]
    return C[m - 1, n - 1]


def _neh_construct(pt, rng, randomize_ties=True):
    """NEH heuristic. Returns (perm, makespan)."""
    n, m = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    if randomize_ties:
        # add tiny random jitter to break ties / perturb ordering
        jitter = rng.random(n) * 1e-9
        order = np.argsort(-(totals + jitter))
    # build sequence incrementally
    seq = [int(order[0])]
    for idx in order[1:]:
        idx = int(idx)
        best_pos = 0
        best_ms = np.inf
        # try inserting at each position
        L = len(seq)
        for pos in range(L + 1):
            new_seq = seq[:pos] + [idx] + seq[pos:]
            ms = _makespan(pt, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [int(idx)] + seq[best_pos:]
    perm = np.array(seq, dtype=np.int64)
    return perm, _makespan(pt, perm)


def _insertion_ls(pt, perm, max_passes=50, deadline=None):
    """Single-job insertion local search. Returns (perm, makespan)."""
    n = len(perm)
    cur = list(perm)
    cur_ms = _makespan(pt, np.array(cur, dtype=np.int64))
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        passes += 1
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            job = cur[i]
            # remove job at i
            rest = cur[:i] + cur[i + 1:]
            best_pos = i
            best_ms = cur_ms
            # try all positions in rest
            for pos in range(len(rest)):
                if pos == i:
                    continue
                new_seq = rest[:pos] + [job] + rest[pos:]
                ms = _makespan(pt, np.array(new_seq, dtype=np.int64))
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_pos = pos
            if best_pos != i or best_ms < cur_ms - 1e-12:
                rest = cur[:i] + cur[i + 1:]
                cur = rest[:best_pos] + [job] + rest[best_pos:]
                cur_ms = best_ms
                improved = True
    return np.array(cur, dtype=np.int64), cur_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]

    start = time.time()
    # safety margin: leave ~5s before the 150s hard kill
    deadline = start + 145.0

    rng = np.random.default_rng(12345)

    best_perm = None
    best_ms = np.inf

    # 1) deterministic NEH (no randomization)
    perm, ms = _neh_construct(pt, rng, randomize_ties=False)
    if ms < best_ms:
        best_ms = ms
        best_perm = perm

    # 2) local search from the NEH solution
    perm2, ms2 = _insertion_ls(pt, best_perm, max_passes=30, deadline=deadline)
    if ms2 < best_ms:
        best_ms = ms2
        best_perm = perm2

    # 3) randomized NEH restarts + local search while time permits
    # Each restart is a fresh NEH with jitter, then a short LS.
    while time.time() < deadline - 10.0:
        perm_r, ms_r = _neh_construct(pt, rng, randomize_ties=True)
        # quick check: only run LS if this start is competitive
        if ms_r <= best_ms + 0.5 * (best_ms if best_ms > 0 else 1.0):
            perm_ls, ms_ls = _insertion_ls(pt, perm_r, max_passes=10, deadline=deadline)
            if ms_ls < best_ms:
                best_ms = ms_ls
                best_perm = perm_ls
        # also consider the raw NEH restart itself
        if ms_r < best_ms:
            best_ms = ms_r
            best_perm = perm_r
        # if we're already very good, allow a deeper LS pass on the best
        if time.time() < deadline - 5.0:
            perm_d, ms_d = _insertion_ls(pt, best_perm, max_passes=5, deadline=deadline)
            if ms_d < best_ms:
                best_ms = ms_d
                best_perm = perm_d

    return list(best_perm)
# EVOLVE-BLOCK-END
