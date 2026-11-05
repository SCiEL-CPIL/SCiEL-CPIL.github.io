"""Flow-shop permutation via NEH + Or-opt/2-opt VND local search + multi-start diversification."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_t, perm):
    """Fast makespan calculation. pt_t: (n_machines, n_jobs) numpy array."""
    n_machines = pt_t.shape[0]
    n = len(perm)
    if n == 0:
        return 0.0
    if n_machines == 0:
        return 0.0

    P = pt_t[:, perm]  # (M, N)
    prev = [0.0] * n_machines
    for k in range(n):
        curr = [0.0] * n_machines
        curr[0] = prev[0] + P[0, k]
        for m in range(1, n_machines):
            v1 = prev[m]
            v2 = curr[m - 1]
            curr[m] = (v1 if v1 > v2 else v2) + P[m, k]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    pt_t = pt.T
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_t, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _bottleneck_initial(pt):
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _or_opt_local_search(pt_t, perm, deadline, max_rounds=500):
    """Or-opt: move segments of length 1, 2, or 3 to a different position."""
    n = len(perm)
    if n <= 3:
        return perm, _makespan_fast(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() > deadline:
                break
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    break
                seg = current[i:i + seg_len]
                rest = current[:i] + current[i + seg_len:]
                for j in range(len(rest) + 1):
                    if time.time() > deadline:
                        break
                    # Skip trivial reinsertion at same spot
                    if j == i or j == i + seg_len:
                        continue
                    test = rest[:j] + seg + rest[j:]
                    ms = _makespan_fast(pt_t, test)
                    if ms < current_ms - 1e-9:
                        current_ms = ms
                        current = test
                        improved = True
                        if current_ms < best_ms - 1e-9:
                            best_ms = current_ms
                            best = list(current)
                        break  # restart for this seg_len
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _swap_local_search(pt_t, perm, deadline, max_rounds=200):
    """Adjacent and non-adjacent swap local search."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test = current[:]
                test[i], test[j] = test[j], test[i]
                ms = _makespan_fast(pt_t, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break  # restart j loop
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _two_opt_local_search(pt_t, perm, deadline, max_rounds=100):
    """2-opt: reverse segments."""
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_t, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
                ms = _makespan_fast(pt_t, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break  # restart j loop
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _vnd(pt_t, perm, deadline):
    """Variable Neighborhood Descent cycling through Or-opt, swap, and 2-opt."""
    current = list(perm)
    current_ms = _makespan_fast(pt_t, current)
    best = list(current)
    best_ms = current_ms

    neighborhoods = [
        (lambda p, d: _or_opt_local_search(pt_t, p, d), 1),
        (lambda p, d: _swap_local_search(pt_t, p, d), 2),
        (lambda p, d: _two_opt_local_search(pt_t, p, d), 3),
    ]

    for _round in range(200):
        if time.time() > deadline:
            break
        improved_any = False
        for ns_func, _ in neighborhoods:
            if time.time() > deadline:
                break
            new_perm, new_ms = ns_func(current, deadline)
            if new_ms < current_ms - 1e-9:
                current = new_perm
                current_ms = new_ms
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
                improved_any = True
                break  # restart cycle from first neighborhood
        if not improved_any:
            break
    return best, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0  # safety margin

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_t = pt.T  # (n_machines, n_jobs)

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _bottleneck_initial(pt),
    ]
    for s in range(4):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan_fast(pt_t, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # VND on best candidate
    if time.time() < deadline - 20.0:
        perm, ms = _vnd(pt_t, best_perm, deadline - 20.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Multi-start: run VND on a few other diverse starts if time permits
    other_candidates = candidates[1:]
    other_candidates.sort(key=lambda c: _makespan_fast(pt_t, c))
    for other in other_candidates[:3]:
        if time.time() < deadline - 15.0:
            perm, ms = _vnd(pt_t, other, deadline - 15.0)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm

    # Final polish with VND
    if time.time() < deadline - 2.0:
        perm, ms = _vnd(pt_t, best_perm, deadline - 1.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
