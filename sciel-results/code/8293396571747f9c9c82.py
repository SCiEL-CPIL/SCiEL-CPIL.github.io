"""Guided local search penalty for TSP.

MODEL-AND-OPTIMIZE: the penalty is a parameterized rule
    P = A_over * over-edge disruption (top-k longest tour edges)
      + A_use  * log(1 + edge_n_used)
where A_over, A_use, and the top-k fraction are the fitted parameters.
They are fit once (offline) against the task's objective by emulating the
evaluator's 20-round loop on random 100-point instances.
"""
import time
import numpy as np

_START = time.perf_counter()
_DEADLINE = _START + 310.0  # safety margin before the 330s kill

# ---------------------------------------------------------------------------
# 2-opt on a given distance matrix (up to max_passes full passes)
# ---------------------------------------------------------------------------
def _two_opt(dist, tour, max_passes=10):
    n = len(tour)
    if n < 4:
        return list(tour)
    t = [int(x) for x in tour]
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = t[i]
            b = t[i + 1]
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                c = t[j]
                d_ = t[(j + 1) % n]
                # delta = (a-c + b-d) - (a-b + c-d)
                if dist[a, c] + dist[b, d_] < dist[a, b] + dist[c, d_]:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return t


def _tour_len(dist, tour):
    t = [int(x) for x in tour]
    return float(sum(dist[t[i], t[i + 1]] for i in range(len(t))))


def _nn_tour(dist):
    n = dist.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = tour[-1]
        best, best_d = -1, float("inf")
        for k in range(n):
            if not visited[k] and dist[cur, k] < best_d:
                best, best_d = k, dist[cur, k]
        tour.append(best)
        visited[best] = True
    return tour


def _emulate(d, init_tour, pen, n_rounds=20, max_passes=10):
    n = d.shape[0]
    usage = np.zeros((n, n))
    best = float("inf")
    cur = list(init_tour)
    for _ in range(n_rounds):
        # add current tour's edges to usage
        for k in range(n):
            a, b = cur[k], cur[(k + 1) % n]
            usage[a, b] += 1
            usage[b, a] += 1
        dm = d + pen(d, cur, usage)
        t1 = _two_opt(dm, cur, max_passes)
        t2 = _two_opt(d, cur, max_passes)
        l1 = _tour_len(d, t1)
        l2 = _tour_len(d, t2)
        if l1 <= l2:
            cur = t1
            cur_len = l1
        else:
            cur = t2
            cur_len = l2
        if cur_len < best:
            best = cur_len
    return best


def _fit_parameters(rng, n_inst=8, n_pts=100, n_rounds=20, max_passes=10):
    """Fit (A_over, A_use, k_frac) to minimize mean best tour length."""
    # pre-generate instances + NN tours (the evaluator's fixed start)
    instances = []
    for _ in range(n_inst):
        pts = rng.random((n_pts, 2))
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 0.0)
        init = _nn_tour(d)
        init = _two_opt(d, init, 10)
        instances.append((d, init))

    cands = [
        (0.10, 0.05, 0.0),
        (0.20, 0.05, 0.0),
        (0.30, 0.05, 0.0),
        (0.50, 0.05, 0.0),
        (0.70, 0.05, 0.0),
        (1.00, 0.05, 0.0),
        (0.20, 0.10, 0.0),
        (0.50, 0.10, 0.0),
        (1.00, 0.10, 0.0),
        (0.30, 0.00, 0.0),
        (0.50, 0.00, 0.0),
        (0.70, 0.00, 0.0),
    ]

    def make_pen(A_over, A_use, k_frac):
        def pen(d, cur, usage):
            n = d.shape[0]
            k = max(1, int(round(k_frac * n)))
            idx = np.arange(n)
            nxt = np.roll(idx, -1)
            tour_len = d[idx, nxt]
            top = np.argsort(tour_len)[-k:]
            P = np.zeros((n, n))
            for t in top:
                a, b = int(idx[t]), int(nxt[t])
                mag = A_over * tour_len[t]
                P[a, b] += mag
                P[b, a] += mag
            if A_use > 0:
                u = np.log1p(np.maximum(usage, 0.0))
                P += A_use * u
            return P
        return pen

    best_c, best_val = None, float("inf")
    for (a, b, k) in cands:
        p = make_pen(a, b, k)
        s = 0.0
        for (d, init) in instances:
            if time.perf_counter() > _DEADLINE - 40:
                break
            s += _emulate(d, init, p, n_rounds, max_passes)
        if s < best_val:
            best_val = s
            best_c = (a, b, k)
        if time.perf_counter() > _DEADLINE - 25:
            break
    if best_c is None:
        return (0.30, 0.10, 0.25)
    return best_c


# ---------------------------------------------------------------------------
# Fit once at import (with a generous, but deadline-bounded, budget)
# ---------------------------------------------------------------------------
try:
    _FIT_RNG = np.random.default_rng(12345)
    _A_OVER, _A_USE, _K_FRAC = _fit_parameters(_FIT_RNG)
except Exception:
    _A_OVER, _A_USE, _K_FRAC = (0.30, 0.10, 0.25)


def _build_penalty(d, tour, usage, A_over, A_use, k_frac):
    """
    Dynamic 2-opt move regret penalty.
    A_over is used as the penalty factor for regret.
    A_use is used as a small global usage penalty.
    k_frac is ignored.
    """
    n = d.shape[0]
    P = np.zeros((n, n))
    
    # Calculate regret for each edge in the tour
    # regret[i] = min delta for 2-opt move breaking edge i
    regret = np.full(n, float('inf'))
    
    t = [int(x) for x in tour]
    for i in range(n):
        u = t[i]
        v = t[(i + 1) % n]
        for j in range(n):
            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                continue
            w = t[j]
            x = t[(j + 1) % n]
            # Check if i and j are non-adjacent in the tour
            # i, i+1, ..., j-1, j
            # If we reverse i+1..j, we break (u,v) and (w,x)
            # New edges: (u,w) and (v,x)
            delta = d[u, w] + d[v, x] - d[u, v] - d[w, x]
            if delta < regret[i]:
                regret[i] = delta
            if delta < regret[j]:
                regret[j] = delta

    # If some regrets are inf (should not happen for N>4), set to 0
    regret = np.where(np.isinf(regret), 0.0, regret)
    
    max_regret = np.max(regret)
    if max_regret < 1e-9:
        max_regret = 1.0
        
    # Penalty: factor * (max_regret - regret[i])
    # Edges with low regret get high penalty
    for i in range(n):
        u = t[i]
        v = t[(i + 1) % n]
        mag = A_over * (max_regret - regret[i])
        P[u, v] += mag
        P[v, u] += mag
        
    if A_use > 0:
        u = np.log1p(np.maximum(usage, 0.0))
        P += A_use * u
        
    return P


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: return the true matrix (a no-perturbation fallback).
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    d = np.asarray(edge_distance, dtype=float)
    n = d.shape[0]
    if n <= 2:
        return d.copy()

    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.shape[0] != n:
        tour = np.arange(n)

    usage = np.asarray(edge_n_used, dtype=float)

    try:
        P = _build_penalty(d, tour, usage, _A_OVER, _A_USE, _K_FRAC)
        result = d + P
        if not np.all(np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
        return result
    except Exception:
        # Never crash: fall back to the true distance matrix.
        return d.copy()
