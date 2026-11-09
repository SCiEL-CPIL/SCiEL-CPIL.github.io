"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_C(perm, pt):
    """Return completion matrix C of shape (n, m): C[j][k] = completion of job perm[j] on machine k.
    C[j][k] = max(C[j][k-1], C[j-1][k]) + pt[perm[j], k], with C[-1][*]=C[*][-1]=0."""
    n, m = pt.shape
    C = np.zeros((n, m), dtype=np.float64)
    for j in range(n):
        job = perm[j]
        prev_row = C[j-1] if j > 0 else None
        c = 0.0
        for k in range(m):
            val = pt[job, k]
            if prev_row is not None and prev_row[k] > c:
                c = prev_row[k]
            c += val
            C[j, k] = c
    return C

def _makespan(perm, pt):
    if len(perm) == 0:
        return 0.0
    n, m = pt.shape
    c = 0.0
    prev = np.zeros(m, dtype=np.float64)
    for j in range(n):
        job = perm[j]
        cur = np.empty(m, dtype=np.float64)
        for k in range(m):
            v = pt[job, k]
            if prev[k] > c:
                c = prev[k]
            c += v
            cur[k] = c
        prev = cur
    return c

def _neh(pt):
    """NEH heuristic: build permutation by inserting jobs in order of decreasing total processing time,
    at the best position (min makespan)."""
    n, m = pt.shape
    order = list(np.argsort(-pt.sum(axis=1)))
    perm = []
    for job in order:
        # try each insertion position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            cand = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(cand, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _recompute_from(C, start, perm, pt):
    """Recompute rows of C from index start onward (inclusive)."""
    n, m = pt.shape
    for j in range(start, n):
        job = perm[j]
        prev_row = C[j-1] if j > 0 else None
        c = 0.0
        for k in range(m):
            v = pt[job, k]
            if prev_row is not None and prev_row[k] > c:
                c = prev_row[k]
            c += v
            C[j, k] = c

def _delta_swap(C, perm, pt, i, j):
    """Return new makespan after swapping positions i and j (i<j), and update C in place.
    We recompute rows from i to n-1. Returns the makespan (C[n-1, m-1])."""
    n, m = pt.shape
    perm[i], perm[j] = perm[j], perm[i]
    _recompute_from(C, i, perm, pt)
    return C[n-1, m-1]

def _swap_back(perm, i, j):
    perm[i], perm[j] = perm[j], perm[i]

def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]

    deadline = time.time() + 140.0  # safety margin under 150s
    rng = random.Random(12345)

    # Initial solution via NEH
    perm = _neh(pt)
    C = _compute_C(perm, pt)
    best_perm = list(perm)
    best_ms = C[n-1, m-1]

    # Simulated Annealing
    T0 = max(best_ms * 0.05, 1e-6)
    T = T0
    cool = 0.9995
    Tmin = 1e-9

    iters = 0
    while T > Tmin and time.time() < deadline:
        # Choose move: mostly adjacent swap, occasionally a random long swap
        if rng.random() < 0.7:
            i = rng.randrange(n - 1)
            j = i + 1
        else:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i

        old_ms = C[n-1, m-1]
        new_ms = _delta_swap(C, perm, pt, i, j)
        delta = new_ms - old_ms

        accept = False
        if delta <= 0:
            accept = True
        else:
            # accept with probability exp(-delta/T)
            if T > 0:
                if rng.random() < np.exp(-delta / T):
                    accept = True
        if accept:
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = list(perm)
        else:
            _swap_back(perm, i, j)
            _recompute_from(C, i, perm, pt)

        T *= cool
        iters += 1
        if iters % 2000 == 0:
            if time.time() > deadline:
                break

    # Final polish: greedy local descent (2-opt adjacent + random) until no improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # adjacent swaps
        for i in range(n - 1):
            old_ms = C[n-1, m-1]
            new_ms = _delta_swap(C, perm, pt, i, i+1)
            if new_ms < old_ms:
                best_ms = new_ms
                best_perm = list(perm)
                improved = True
            else:
                _swap_back(perm, i, i+1)
                _recompute_from(C, i, perm, pt)
        # a few random long swaps
        for _ in range(3):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            old_ms = C[n-1, m-1]
            new_ms = _delta_swap(C, perm, pt, i, j)
            if new_ms < old_ms:
                best_ms = new_ms
                best_perm = list(perm)
                improved = True
            else:
                _swap_back(perm, i, j)
                _recompute_from(C, i, perm, pt)

    return best_perm
# EVOLVE-BLOCK-END
