"""Flow-shop permutation construction: NEH + simulated annealing + Variable Neighborhood Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _row_for_job(job, prev_row, P):
    """Compute the completion-time row for `job` given the previous row."""
    n_machines = P.shape[1]
    row = np.empty(n_machines)
    row[0] = prev_row[0] + P[job, 0]
    if n_machines > 1:
        for m in range(1, n_machines):
            row[m] = (prev_row[m] if prev_row[m] > row[m - 1] else row[m - 1]) + P[job, m]
    return row


def _build_C(order, P):
    """Build the full completion-time matrix for `order`."""
    n_jobs = len(order)
    n_machines = P.shape[1]
    C = np.empty((n_jobs, n_machines))
    C[0] = np.cumsum(P[order[0]])
    for idx in range(1, n_jobs):
        C[idx] = _row_for_job(order[idx], C[idx - 1], P)
    return C


def _makespan(order, P):
    """Compute makespan for a given job order."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    if P.shape[1] == 0:
        return 0.0
    C = _build_C(order, P)
    return C[-1, -1]


def _recompute_suffix(C, order, P, start):
    """Recompute completion rows from index `start` onward; returns new makespan."""
    n_jobs = len(order)
    n_machines = P.shape[1]
    for idx in range(start, n_jobs):
        prev = C[idx - 1] if idx > 0 else None
        if prev is None:
            C[idx] = np.cumsum(P[order[idx]])
        else:
            C[idx] = _row_for_job(order[idx], prev, P)
    return C[-1, -1]


def _neh_heuristic(P, sort_key='total', deadline=None):
    """NEH heuristic: sort by key, insert each job at its best position."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    if sort_key == 'total':
        key_vals = P.sum(axis=1)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
    elif sort_key == 'first':
        key_vals = P[:, 0]
    else:
        key_vals = P.sum(axis=1)
    sorted_indices = np.argsort(-key_vals)

    current_seq = [int(sorted_indices[0])]
    for k in range(1, n_jobs):
        if deadline is not None and time.time() > deadline:
            break
        job = int(sorted_indices[k])
        best_pos = len(current_seq)
        best_makespan = float('inf')
        for pos in range(len(current_seq) + 1):
            if deadline is not None and pos % 8 == 0 and time.time() > deadline:
                break
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    return current_seq


def _simulated_annealing(order, P, deadline, rng):
    """Simulated annealing to escape shallow local minima."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_C = _build_C(current_order, P)
    current_ms = current_C[-1, -1]
    best_order = list(current_order)
    best_ms = current_ms

    start = time.time()
    temp = max(current_ms * 0.15, 1.0)
    cooling = 0.99

    while temp > 1e-6 and time.time() < deadline:
        i = int(rng.integers(0, n_jobs))
        j = int(rng.integers(0, n_jobs))
        if i == j:
            continue
        # apply swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_ms = _recompute_suffix(current_C, current_order, P, min(i, j))
        delta = new_ms - current_ms
        if delta < 0 or (temp > 1e-9 and rng.random() < np.exp(-delta / temp)):
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_order = list(current_order)
        else:
            # revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            _recompute_suffix(current_C, current_order, P, min(i, j))
        temp *= cooling

    return best_order


def _vns_search(order, P, deadline, rng, max_rounds=40):
    """Variable Neighborhood Search: shake + descent through N1, N2, N3."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_C = _build_C(current_order, P)
    current_ms = current_C[-1, -1]
    best_order = list(current_order)
    best_ms = current_ms

    def _descend(seq, C, ms):
        """Descent: cycle N1, N2, N3 until no improvement; returns (seq, C, ms)."""
        n2_full = n_jobs <= 120
        for _ in range(3):
            improved = False
            # N1: adjacent swap (best improvement)
            for i in range(n_jobs - 1):
                if deadline is not None and time.time() > deadline:
                    return seq, C, ms
                j1 = seq[i]
                j2 = seq[i + 1]
                if i == 0:
                    new_row_i = np.cumsum(P[j2])
                else:
                    new_row_i = _row_for_job(j2, C[i - 1], P)
                new_row_i1 = _row_for_job(j1, new_row_i, P)
                cand = new_row_i1[-1]
                if cand < ms - 1e-12:
                    seq[i], seq[i + 1] = j2, j1
                    C[i] = new_row_i
                    C[i + 1] = new_row_i1
                    ms = _recompute_suffix(C, seq, P, i + 2)
                    improved = True
                    break
            if improved or deadline is not None and time.time() > deadline:
                continue
            # N2: 2-opt block reversal
            for i in range(n_jobs - 1):
                if deadline is not None and time.time() > deadline:
                    return seq, C, ms
                if not n2_full:
                    if i % 3 != 0:
                        continue
                for j in range(i + 1, n_jobs):
                    if deadline is not None and j % 8 == 0 and time.time() > deadline:
                        return seq, C, ms
                    if not n2_full and (j - i) > 8:
                        break
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    cand = _recompute_suffix(C, seq, P, i)
                    if cand < ms - 1e-12:
                        ms = cand
                        improved = True
                        break
                    else:
                        seq[i:j + 1] = seq[i:j + 1][::-1]
                if improved:
                    break
            if improved or deadline is not None and time.time() > deadline:
                continue
            # N3: 3-opt segment relocation (sampled)
            n_moves = 40 if n_jobs <= 60 else 15
            for _ in range(n_moves):
                if deadline is not None and time.time() > deadline:
                    return seq, C, ms
                a = int(rng.integers(0, n_jobs))
                b = int(rng.integers(0, n_jobs))
                if a > b:
                    a, b = b, a
                seg_len = b - a + 1
                if seg_len >= n_jobs:
                    continue
                d = int(rng.integers(0, n_jobs - seg_len + 1))
                if a <= d <= b:
                    continue
                seg = seq[a:b + 1]
                rest = seq[:a] + seq[b + 1:]
                if d < a:
                    new_seq = rest[:d] + seg + rest[d:]
                else:
                    new_seq = rest[:d] + seg + rest[d:]
                C2 = _build_C(new_seq, P)
                cand = C2[-1, -1]
                if cand < ms - 1e-12:
                    seq[:] = new_seq
                    C[:] = C2
                    ms = cand
                    improved = True
                    break
            if improved or deadline is not None and time.time() > deadline:
                continue
            if not improved:
                break
        return seq, C, ms

    for _round in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        # Shake: random perturbation, larger as search stalls
        for k in range(3):
            i = int(rng.integers(0, n_jobs))
            j = int(rng.integers(0, n_jobs))
            if i == j:
                continue
            current_order[i], current_order[j] = current_order[j], current_order[i]
        current_ms = _recompute_suffix(current_C, current_order, P, 0)

        current_order, current_C, current_ms = _descend(current_order, current_C, current_ms)

        if current_ms < best_ms - 1e-12:
            best_ms = current_ms
            best_order = list(current_order)
        if deadline is not None and time.time() > deadline:
            break

    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    NEH variants -> simulated annealing -> Variable Neighborhood Search.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if P.shape[1] == 0:
        return list(range(n_jobs))

    start_time = time.time()
    deadline = start_time + 145.0
    rng = np.random.default_rng(12345)

    # --- NEH candidates ---
    candidates = []
    candidates.append(_neh_heuristic(P, 'total', deadline))
    if time.time() < deadline - 10.0:
        candidates.append(_neh_heuristic(P, 'first_last', deadline))
    if time.time() < deadline - 10.0:
        candidates.append(_neh_heuristic(P, 'first', deadline))

    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    for seq in candidates[1:]:
        ms = _makespan(seq, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = seq

    # --- Simulated annealing ---
    if time.time() < deadline - 15.0:
        sa_seq = _simulated_annealing(best_seq, P, deadline - 10.0, rng)
        sa_ms = _makespan(sa_seq, P)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_seq = sa_seq

    # --- Variable Neighborhood Search ---
    if time.time() < deadline - 2.0:
        vns_seq = _vns_search(best_seq, P, deadline, rng, max_rounds=40)
        vns_ms = _makespan(vns_seq, P)
        if vns_ms < best_ms:
            best_ms = vns_ms
            best_seq = vns_seq

    # Final validation
    if len(best_seq) != n_jobs or sorted(best_seq) != list(range(n_jobs)):
        best_seq = list(range(n_jobs))

    return list(best_seq)
# EVOLVE-BLOCK-END
