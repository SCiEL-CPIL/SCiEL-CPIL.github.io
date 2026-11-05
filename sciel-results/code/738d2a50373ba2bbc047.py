# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _makespan_vec(pt, perm):
    """Vectorized flow-shop makespan.

    pt:  (n_jobs, m) array
    perm: (n,) int array of job indices
    Returns scalar makespan.
    """
    n = perm.size
    if n == 0:
        return 0.0
    P = pt[perm]                      # (n, m)
    if P.shape[1] == 1:
        return float(P[:, 0].sum())
    C = np.empty_like(P)
    C[:, 0] = P[:, 0]
    np.cumsum(C[:, 0], axis=0, out=C[:, 0])
    prev = C[0]
    for i in range(1, n):
        np.maximum(prev, C[i - 1], out=prev)
        prev += P[i]
        C[i] = prev
    return float(C[-1, -1])


def _build_cum(pt):
    """Per-machine cumulative processing-time prefix sums (n+1, m)."""
    n, m = pt.shape
    cum = np.empty((n + 1, m), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(pt, axis=0, out=cum[1:])
    return cum


def _insert_delta(cum, m, i, j):
    """Exact makespan change for moving the job at index i to index j (j<i).

    Returns (delta, moved_job). O(m) using per-machine prefix sums.
    """
    if i <= j:
        return 0.0, -1
    job = int(i)
    a = i - 1
    b = j
    # For each machine, the moved job's new start = max(machine cum at b,
    # machine completion at a).  Old completion on the last machine was
    # cum[n-1, m-1] (job was last).
    start = np.maximum(cum[b], cum[a])
    new_last = float(start[-1] + cum[n - 1, -1] - cum[a, -1])
    return new_last - float(cum[n - 1, -1]), job


def _neh(pt_list, pt):
    """NEH heuristic (descending total processing)."""
    n = pt.shape[0]
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    for job in order[1:]:
        best_pos, best_ms = 0, float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_vec(pt, np.asarray(test, dtype=np.int64))
            if ms < best_ms:
                best_ms, best_pos = ms, pos
        perm.insert(best_pos, job)
    return perm


def _palmer(pt):
    n, m = pt.shape
    scores = np.empty(n, dtype=np.float64)
    for j in range(n):
        s = 0.0
        for k in range(m):
            if k < m // 2:
                s += (m - 1 - k) * pt[j, k]
            else:
                s -= k * pt[j, k]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt(pt):
    return list(np.argsort(-pt.sum(axis=1)))


def _spt(pt):
    return list(np.argsort(pt.sum(axis=1)))


def _bottleneck(pt):
    loads = pt.sum(axis=0)
    b = int(np.argmax(loads))
    return list(np.argsort(-pt[:, b]))


def _mean(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first(pt):
    return list(np.argsort(-pt[:, 0]))


def _last(pt):
    return list(np.argsort(-pt[:, -1]))


def _random(n, rng):
    p = list(range(n))
    rng.shuffle(p)
    return p


def _insertion_ls(pt, cum, perm, deadline, tol=1e-9, max_rounds=40000):
    """Local search over insertion moves using the O(m) exact delta."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_vec(pt, np.asarray(perm, dtype=np.int64))
    arr = np.asarray(perm, dtype=np.int64)
    cur = _makespan_vec(pt, arr)
    best = list(perm)
    best_ms = cur
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i):
                d, _ = _insert_delta(cum, arr.shape[1], i, j)
                if d < -tol:
                    arr = np.concatenate((arr[:j], arr[i:i + 1], arr[j:i]))
                    cur += d
                    improved = True
                    if cur < best_ms - tol:
                        best_ms = cur
                        best = arr.tolist()
                    break
            if improved:
                break
        if not improved:
            break
    if best_ms < cur:
        return best, best_ms
    return arr.tolist(), cur


def _two_opt_ls(pt, cum, perm, deadline, tol=1e-9, max_rounds=20000):
    """Local search over segment reversals (full recompute, exact)."""
    n = len(perm)
    if n <= 2:
        return perm, _makespan_vec(pt, np.asarray(perm, dtype=np.int64))
    arr = np.asarray(perm, dtype=np.int64)
    cur = _makespan_vec(pt, arr)
    best = list(perm)
    best_ms = cur
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                d = j - i + 1
                arr = np.concatenate((arr[:i], arr[i:j + 1][::-1], arr[j + 1:]))
                ms = _makespan_vec(pt, arr)
                if ms < cur - tol:
                    cur = ms
                    improved = True
                    if cur < best_ms - tol:
                        best_ms = cur
                        best = arr.tolist()
                    break
            if improved:
                break
        if not improved:
            break
    if best_ms < cur:
        return best, best_ms
    return arr.tolist(), cur


def _simulated_annealing(pt, cum, perm, deadline, seed=7, tol=1e-9):
    """SA with insertion moves, using the O(m) delta for acceptance."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_vec(pt, np.asarray(perm, dtype=np.int64))
    arr = np.asarray(perm, dtype=np.int64)
    cur = _makespan_vec(pt, arr)
    best = list(perm)
    best_ms = cur
    T = max(cur * 0.05, 1.0)
    T_min = max(cur * 0.0005, 0.01)
    cool = 0.9995
    while time.time() < deadline:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if j >= i:
            j = i - 1
        if j < 0:
            continue
        d, _ = _insert_delta(cum, arr.shape[1], i, j)
        if d < 0 or rng.random() < np.exp(-d / T):
            arr = np.concatenate((arr[:j], arr[i:i + 1], arr[j:i]))
            cur += d
            if cur < best_ms:
                best_ms = cur
                best = arr.tolist()
        T *= cool
        if T < T_min:
            T = T_min
    if best_ms < cur:
        return best, best_ms
    return arr.tolist(), cur


def _order_crossover(a, b, rng):
    """Order crossover (OX) on two permutations."""
    n = len(a)
    if n <= 2:
        return list(a)
    i = rng.randrange(n)
    j = rng.randrange(n)
    if i > j:
        i, j = j, i
    if j - i < 2:
        return list(a)
    child = [None] * n
    for k in range(i, j + 1):
        child[k] = a[k]
    seg = set(a[i:j + 1])
    t = 0
    for val in b:
        if val not in seg:
            if child[t] is None:
                child[t] = val
            t += 1
            if t >= n:
                break
    for k in range(n):
        if child[k] is None:
            for val in b:
                if child.count(val) < b.count(val):
                    child[k] = val
                    break
    return child


def _ga(pt, cum, pop, deadline, tol=1e-9):
    """Small GA with order crossover + insertion mutation + selection."""
    rng = random.Random(2024)
    pop = [np.asarray(p, dtype=np.int64) for p in pop]
    ms = [_makespan_vec(pt, p) for p in pop]
    order = np.argsort(ms)
    best = pop[order[0]]
    best_ms = ms[order[0]]
    n = len(pop[0])
    while time.time() < deadline:
        # Elitism: keep top half
        half = max(2, len(pop) // 2)
        keep = [pop[k] for k in order[:half]]
        new_pop = list(keep)
        while len(new_pop) < len(pop) and time.time() < deadline:
            pa = keep[rng.randrange(half)]
            pb = keep[rng.randrange(half)]
            child = _order_crossover(pa.tolist(), pb.tolist(), rng)
            # insertion mutation
            if n > 1 and rng.random() < 0.7:
                i = rng.randrange(n)
                j = rng.randrange(n)
                if j >= i:
                    j = i - 1
                if j >= 0:
                    child = child[:j] + child[i:i + 1] + child[j:i]
            c = np.asarray(child, dtype=np.int64)
            m = _makespan_vec(pt, c)
            new_pop.append(c)
            if m < best_ms:
                best_ms = m
                best = c
        pop = new_pop
        ms = [_makespan_vec(pt, p) for p in pop]
        order = np.argsort(ms)
    return best.tolist(), best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 0:
        return list(range(n))

    start = time.time()
    deadline = start + 145.0
    cum = _build_cum(pt)

    # Diverse initial solutions (incl. dynamic bottleneck heuristic)
    rng = random.Random(0)
    inits = [
        _neh(pt.tolist(), pt),
        _palmer(pt),
        _lpt(pt),
        _spt(pt),
        _bottleneck(pt),
        _mean(pt),
        _first(pt),
        _last(pt),
    ]
    for s in range(6):
        inits.append(_random(n, random.Random(100 + s)))

    scored = []
    for cand in inits:
        scored.append((_makespan_vec(pt, np.asarray(cand, dtype=np.int64)), cand))
    scored.sort(key=lambda x: x[0])

    best_ms = scored[0][0]
    best_perm = scored[0][1]

    # Polish top candidates with insertion local search (fast O(m) delta)
    for ms0, cand in scored:
        if time.time() > deadline - 25.0:
            break
        if ms0 > best_ms * 1.25:
            continue
        perm, ms = _insertion_ls(pt, cum, cand, deadline - 25.0)
        if ms < best_ms:
            best_ms, best_perm = ms, perm

    # 2-opt polish on the best
    if time.time() < deadline - 18.0:
        perm, ms = _two_opt_ls(pt, cum, best_perm, deadline - 18.0)
        if ms < best_ms:
            best_ms, best_perm = ms, perm

    # Simulated annealing to escape local optima
    if time.time() < deadline - 12.0:
        perm, ms = _simulated_annealing(pt, cum, best_perm, deadline - 12.0, seed=7)
        if ms < best_ms:
            best_ms, best_perm = ms, perm

    # Genetic algorithm with order crossover on a diverse population
    if time.time() < deadline - 6.0:
        pop = [best_perm]
        for _, cand in scored[:5]:
            if len(pop) >= 12:
                break
            if cand not in pop:
                pop.append(cand)
        while len(pop) < 12:
            pop.append(_random(n, random.Random(500 + len(pop))))
        perm, ms = _ga(pt, cum, pop, deadline - 6.0)
        if ms < best_ms:
            best_ms, best_perm = ms, perm

    # Final high-precision polish (tighter tolerance, more iterations)
    if time.time() < deadline - 2.0:
        perm, ms = _insertion_ls(pt, cum, best_perm, deadline - 1.0,
                                 tol=1e-12, max_rounds=100000)
        if ms < best_ms:
            best_ms, best_perm = ms, perm
    if time.time() < deadline - 0.5:
        perm, ms = _two_opt_ls(pt, cum, best_perm, deadline - 0.5,
                               tol=1e-12, max_rounds=30000)
        if ms < best_ms:
            best_ms, best_perm = ms, perm

    return best_perm
# EVOLVE-BLOCK-END
