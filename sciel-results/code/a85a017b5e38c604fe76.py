"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _deadline_guard():
    """Return remaining seconds before the 150s wall-clock kill (with safety margin)."""
    # Safety margin of 3 seconds so we never get killed mid-evaluation.
    return 150.0 - 3.0 - (time.time() - _START_TIME)


def _makespan(order, p, n_jobs, n_machines):
    """Exact makespan for a given job order (list of job indices)."""
    if n_jobs == 0:
        return 0.0
    # completion[m] = completion time of machine m for the last job processed
    # We simulate the flow shop.
    completion = np.zeros(n_machines)
    for j in order:
        t = 0.0
        for m in range(n_machines):
            t = max(t, completion[m]) + p[j, m]
            completion[m] = t
    return float(completion[-1])


def _makespan_fast(order, p, n_jobs, n_machines):
    """Vectorized makespan: completion[m] = max over jobs in order of (sum of p[j, :m+1] ... ).
    Use the classic formula: C_jm = max(C_j,m-1, C_j-1,m) + p_jm.
    We compute per-machine running max of partial sums.
    """
    # For each machine m, the completion time after processing the whole sequence is:
    # C_m = max over k ( sum_{i<=k} p_{order_i, m} + sum_{i<k} p_{order_i, m+1...} )? 
    # Simpler robust vectorized approach using cumulative sums per machine and a
    # running max across machines.
    # Build matrix P_seq = p[order]  (n_jobs x n_machines)
    P = p[order]
    # cumulative along jobs for each machine
    # completion[m] = max_{j} ( prefixsum_m[j] + max_{m'>m} (prefixsum_{m'}[j-1]) )
    # We'll just do the O(n*m) loop but it's fine for moderate sizes; provide vectorized
    # fallback below.
    completion = np.zeros(n_machines)
    for j in range(n_jobs):
        row = P[j]
        t = 0.0
        for m in range(n_machines):
            if t < completion[m]:
                t = completion[m]
            t += row[m]
            completion[m] = t
    return float(completion[-1])


def _neh_initialization(p, n_jobs, n_machines, rng):
    """NEH heuristic: insert jobs one by one (sorted by total processing time)
    at the position that minimizes the resulting makespan (incremental)."""
    totals = p.sum(axis=1)
    order_by_total = np.argsort(-totals)
    # optional small randomization of the sort to create diverse NEH variants
    seq = list(order_by_total)
    # build sequence incrementally
    current = []
    # completion times array for current sequence
    # We recompute makespan for each candidate insertion position (n+1 positions).
    for j in seq:
        best_pos = 0
        best_mk = np.inf
        n = len(current)
        for pos in range(n + 1):
            cand = current[:pos] + [j] + current[pos:]
            mk = _makespan(cand, p, len(cand), n_machines)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        current = current[:best_pos] + [j] + current[best_pos:]
    return current


def _v_shaped_initialization(p, n_jobs, n_machines, rng):
    """V-shaped ordering: largest totals in front, smallest in back (reversed)."""
    if n_jobs <= 2:
        return list(range(n_jobs))
    totals = p.sum(axis=1)
    sorted_idx = np.argsort(-totals)
    half = n_jobs // 2
    front = list(sorted_idx[:half])
    back = list(sorted_idx[half:])
    back.reverse()
    return front + back


def _random_initialization(n_jobs, rng):
    perm = np.arange(n_jobs)
    rng.shuffle(perm)
    return list(perm)


def _local_search_insertion(order, p, n_jobs, n_machines, rng, max_iters=200, tabu_len=8):
    """Best-improvement insertion local search with short tabu memory."""
    current = list(order)
    current_mk = _makespan(current, p, n_jobs, n_machines)
    tabu = []
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        best_mk = current_mk
        best_move = None
        n = len(current)
        # candidate: remove i, insert at j (j != i, j != i-1 to avoid no-op)
        for i in range(n):
            if i in tabu:
                continue
            j = current.pop(i)
            for pos in range(n):
                if pos == i or pos == i - 1:
                    continue
                cand = current[:pos] + [j] + current[pos:]
                mk = _makespan(cand, p, n, n_machines)
                if mk < best_mk - 1e-12:
                    best_mk = mk
                    best_move = (i, pos)
            current.insert(i, j)
        if best_move is not None:
            i, pos = best_move
            j = current.pop(i)
            current = current[:pos] + [j] + current[pos:]
            current_mk = best_mk
            tabu.append(i)
            if len(tabu) > tabu_len:
                tabu.pop(0)
            improved = True
    return current, current_mk


def _two_opt(order, p, n_jobs, n_machines, max_iters=50):
    """2-opt (reverse a segment) local search."""
    current = list(order)
    current_mk = _makespan(current, p, n_jobs, n_machines)
    n = len(current)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        best_mk = current_mk
        best_move = None
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = current[:i] + list(reversed(current[i:j+1])) + current[j+1:]
                mk = _makespan(cand, p, n, n_machines)
                if mk < best_mk - 1e-12:
                    best_mk = mk
                    best_move = (i, j)
        if best_move is not None:
            i, j = best_move
            current = current[:i] + list(reversed(current[i:j+1])) + current[j+1:]
            current_mk = best_mk
            improved = True
    return current, current_mk


_START_TIME = time.time()


def order_jobs(processing_times):
    global _START_TIME
    _START_TIME = time.time()

    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    rng = np.random.default_rng(12345)

    # Build a pool of diverse initial solutions.
    inits = []
    # NEH (deterministic)
    inits.append(_neh_initialization(p, n_jobs, n_machines, rng))
    # V-shaped
    inits.append(_v_shaped_initialization(p, n_jobs, n_machines, rng))
    # A few random
    n_random = 3
    for _ in range(n_random):
        inits.append(_random_initialization(n_jobs, rng))

    best_order = None
    best_mk = np.inf

    # Evaluate and improve each initial solution.
    for init in inits:
        if _deadline_guard() <= 1.0:
            break
        order, mk = _local_search_insertion(init, p, n_jobs, n_machines, rng, max_iters=150, tabu_len=6)
        # polish with 2-opt if time permits
        if _deadline_guard() > 5.0:
            order, mk = _two_opt(order, p, n_jobs, n_machines, max_iters=30)
        if mk < best_mk:
            best_mk = mk
            best_order = list(order)

    # If we still have time, do additional restarts from perturbed best.
    restarts = 0
    while _deadline_guard() > 8.0 and restarts < 40:
        restarts += 1
        # perturb best_order: random swap of two positions
        n = len(best_order)
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        perturbed = list(best_order)
        perturbed[a], perturbed[b] = perturbed[b], perturbed[a]
        order, mk = _local_search_insertion(perturbed, p, n_jobs, n_machines, rng, max_iters=120, tabu_len=6)
        if _deadline_guard() > 5.0:
            order, mk = _two_opt(order, p, n_jobs, n_machines, max_iters=20)
        if mk < best_mk:
            best_mk = mk
            best_order = list(order)

    if best_order is None:
        # fallback: NEH without local search
        best_order = _neh_initialization(p, n_jobs, n_machines, rng)

    return list(best_order)
# EVOLVE-BLOCK-END
