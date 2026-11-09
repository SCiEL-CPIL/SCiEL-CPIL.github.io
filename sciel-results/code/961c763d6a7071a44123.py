"""Flow-shop makespan minimizer: NEH heuristic + guided local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(pt, order):
    """Compute makespan for a given job order. order is an index array."""
    n = len(order)
    if n == 0:
        return 0.0
    sel = pt[order]
    m = sel.shape[1]
    # C[m, n] completion times per machine/job-position
    C = np.empty((m, n), dtype=np.float64)
    C[0, 0] = sel[0, 0]
    if n > 1:
        C[0, 1:] = C[0, :-1] + sel[0, 1:]
    if m > 1:
        # C[i,0] = C[i-1,0] + sel[i,0]
        C[1:, 0] = C[:-1, 0] + sel[1:, 0]
        # C[i,j] = max(C[i-1,j], C[i,j-1]) + sel[i,j]
        for i in range(1, m):
            row = C[i, 1:]
            prev = C[i - 1, 1:]
            left = C[i, :-1]
            np.maximum(prev, left, out=row)
            row += sel[i, 1:]
    return float(C[-1, -1])


def _neh(pt, rng):
    """NEH (Nawaz, Enscore, Ham) construction heuristic."""
    n = pt.shape[0]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    seq = [int(order[0])]
    for j in order[1:]:
        j = int(j)
        best_pos, best_ms = 0, None
        for p in range(len(seq) + 1):
            cand = seq[:p] + [j] + seq[p:]
            ms = _makespan(pt, cand)
            if best_ms is None or ms < best_ms:
                best_ms = ms
                best_pos = p
        seq.insert(best_pos, j)
    return seq


def _insertion(pt, seq, rng, deadline, best_seq, best_ms):
    """Guided insertion local search. Returns (best_seq, best_ms, improved)."""
    n = len(seq)
    improved = False
    while time.time() < deadline:
        progress = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = seq[i]
            for p in range(n + 1):
                if p == i:
                    continue
                cand = seq[:i] + seq[i + 1:]
                cand.insert(p, job)
                ms = _makespan(pt, cand)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_seq = cand
                    progress = True
                    improved = True
        if not progress:
            break
    return best_seq, best_ms, improved


def _twopt(pt, seq, rng, deadline, best_seq, best_ms):
    """2-opt (segment reversal) local search."""
    n = len(seq)
    improved = False
    while time.time() < deadline:
        progress = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                ms = _makespan(pt, cand)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_seq = cand
                    progress = True
                    improved = True
        if not progress:
            break
    return best_seq, best_ms, improved


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]

    start = time.time()
    # Safety margin: return well before the 150s kill (and 120s eval limit).
    deadline = start + 110.0

    rng = np.random.default_rng(12345)

    # Initial solution via NEH.
    best_seq = _neh(pt, rng)
    best_ms = _makespan(pt, best_seq)

    # Local search phases with time checks.
    best_seq, best_ms, _ = _insertion(pt, list(best_seq), rng, deadline, best_seq, best_ms)
    best_seq, best_ms, _ = _twopt(pt, list(best_seq), rng, deadline, best_seq, best_ms)
    best_seq, best_ms, _ = _insertion(pt, list(best_seq), rng, deadline, best_seq, best_ms)

    # Optional: a few random-restart NEH + local search while time permits.
    attempts = 0
    while time.time() < deadline and attempts < 40:
        attempts += 1
        cand_seq = _neh(pt, rng)
        cand_ms = _makespan(pt, cand_seq)
        if cand_ms < best_ms - 1e-9:
            best_seq = cand_seq
            best_ms = cand_ms
            best_seq, best_ms, _ = _insertion(pt, list(best_seq), rng, deadline, best_seq, best_ms)
            best_seq, best_ms, _ = _twopt(pt, list(best_seq), rng, deadline, best_seq, best_ms)

    return [int(x) for x in best_seq]
# EVOLVE-BLOCK-END
