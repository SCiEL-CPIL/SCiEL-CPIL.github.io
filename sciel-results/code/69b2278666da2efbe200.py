"""Flow-shop permutation construction: NEH construction + insertion local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(pt, perm):
    """Makespan of a given job permutation (list of job indices)."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # C[i][k] = completion of job perm[i] on machine k
    C = np.empty((n, m), dtype=float)
    first = pt[perm[0]]
    C[0] = np.cumsum(first)
    for i in range(1, n):
        row = pt[perm[i]]
        C[i, 0] = C[i - 1, 0] + row[0]
        C[i, 1:] = np.maximum(C[i - 1, 1:], C[i, :-1]) + row[1:]
    return float(C[n - 1, m - 1])


def _best_insertion(pt, perm, i, m, n):
    """Best single re-insertion of job perm[i]; returns (gain, new_perm)."""
    job = perm[i]
    base = _makespan(pt, perm)
    rest = perm[:i] + perm[i + 1:]
    best_gain = 0.0
    best_new = None
    for p in range(n):
        cand = rest[:p] + [job] + rest[p:]
        ms = _makespan(pt, cand)
        gain = base - ms
        if gain > best_gain:
            best_gain = gain
            best_new = cand
    return best_gain, best_new


def order_jobs(processing_times):
    t0 = time.time()
    deadline = 150.0 - 5.0  # safety margin before the 150s kill
    pt = np.asarray(processing_times, dtype=float)
    n, m = pt.shape
    if n <= 1:
        return list(range(n))

    # --- Stage 1: NEH construction (insert each job in best position) ---
    order = list(np.argsort(-pt.sum(axis=1)))
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for p in range(len(seq) + 1):
            cand = seq[:p] + [j] + seq[p:]
            ms = _makespan(pt, cand)
            if ms < best_ms:
                best_ms = ms
                best_pos = p
        seq = seq[:best_pos] + [j] + seq[best_pos:]

    best_perm = seq
    best_ms = _makespan(pt, best_perm)

    # --- Stage 2: first-improvement insertion local search ---
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        for i in range(n):
            if (time.time() - t0) > deadline:
                break
            gain, new_perm = _best_insertion(pt, best_perm, i, m, n)
            if gain > 1e-12:
                best_perm = new_perm
                best_ms = _makespan(pt, best_perm)
                improved = True

    return list(best_perm)
# EVOLVE-BLOCK-END
