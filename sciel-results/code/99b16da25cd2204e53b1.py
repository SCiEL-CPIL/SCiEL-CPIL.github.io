# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_C(a, b, p):
    """C[i] = sum_j a[i,j] * b[p[i], p[j]]"""
    B = b[np.ix_(p, p)]
    return (a @ B).astype(np.int64)


def _swap_delta(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j.
    Vectorized O(n) operation (dot product).
    """
    pi = p[i]
    pj = p[j]
    
    # w_k = a[i, k] - a[j, k] + a[k, i] - a[k, j]
    w = a[i, :] - a[j, :] + a[:, i] - a[:, j]
    # d_k = b[pj, k] - b[pi, k]
    d = b[pj, :] - b[pi, :]
    
    # Sum over all k
    total_dot = int(np.dot(w, d))
    
    # Subtract k=i and k=j terms
    wi = 2 * a[i, i] - a[j, i] - a[i, j]
    di = b[pj, pi] - b[pi, pi]
    
    wj = a[i, j] + a[j, i] - 2 * a[j, j]
    dj = b[pj, pj] - b[pi, pj]
    
    cross_sum = total_dot - int(wi * di) - int(wj * dj)
    
    # Direct terms
    direct = (a[i, i] * (b[pj, pj] - b[pi, pi]) + 
              a[j, j] * (b[pi, pi] - b[pj, pj]) + 
              a[i, j] * (b[pj, pi] - b[pi, pj]) + 
              a[j, i] * (b[pi, pj] - b[pj, pi]))
    
    return cross_sum + direct


def _local_search(a, b, p, deadline, best_cost, best_p, max_passes=50):
    """First-improvement 2-opt local search with time guard."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    passes = 0
    while time.time() < deadline and passes < max_passes:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break  # first improvement: restart inner loop
            if time.time() >= deadline:
                break
        passes += 1
        if not improved:
            break
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=100.0, t_end=1.0, alpha=0.99):
    """Simulated annealing polish."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    t = t_start
    while time.time() < deadline and t > t_end:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        t *= alpha
    
    # Final local search
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill

    rng = np.random.default_rng(12345)
    
    # --- Constructive starts: multiple strategies ---
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Strategy 1: rank matching (heavy facilities to central locations)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    c1 = _qap_cost(a, b, p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
    
    # Strategy 2: random
    p2 = rng.permutation(n).astype(np.int64)
    c2 = _qap_cost(a, b, p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()
    
    # Strategy 3: greedy nearest-neighbor style
    p3 = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    order = np.argsort(-a_row_sums, kind="stable")
    for idx in order:
        # Pick location minimizing sum of distances to already placed heavy facilities
        if not np.any(used):
            p3[idx] = int(np.argmin(b_col_sums))
        else:
            placed = np.where(~used)[0]
            # score for each unused location: weighted by flows to placed
            scores = np.zeros(n, dtype=np.float64)
            for j in placed:
                scores += a[idx, j] * b[:, p3[j]] + a[j, idx] * b[p3[j], :]
            candidates = np.where(~used)[0]
            p3[idx] = int(candidates[np.argmin(scores[candidates])])
        used[p3[idx]] = True
    c3 = _qap_cost(a, b, p3)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()
    
    # --- Local search from best start ---
    p = best_p.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)
    
    # --- Simulated annealing from current best ---
    if time.time() < deadline - 5.0:
        p = best_p.copy()
        best_cost, best_p = _anneal(a, b, p, deadline, best_cost, best_p, rng)
    
    # --- Additional local search passes if time remains ---
    if time.time() < deadline - 2.0:
        p = best_p.copy()
        best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)
    
    # Final verification
    final_cost = _qap_cost(a, b, best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
