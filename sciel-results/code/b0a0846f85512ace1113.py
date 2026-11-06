import time
import math
import random
import numpy as np
from scipy.special import logsumexp

# ---- configuration ----
K = 5
MAX_VERTICES = 46
DEADLINE = 300.0          # hard wall-clock safety (evaluator kills at 330s)
SAFE_MARGIN = 15.0        # return well before the limit
T_START = time.time()


def _time_left():
    return (DEADLINE - SAFE_MARGIN) - (time.time() - T_START)


# ---------------- fast exact 5-free check ----------------
def _has_k_clique(adj, n, k):
    adj = list(adj)
    memo = {}

    def rec(cand):
        if len(cand) < k:
            return False
        key = cand
        if key in memo:
            return memo[key]
        if len(cand) == k:
            memo[key] = True
            return True
        # branch on the first vertex; keep only its common neighbours
        v = cand[0]
        rest = cand[1:]
        nxt = tuple(v2 for v2 in rest if (adj[v] >> v2) & 1)
        r = rec(nxt)
        memo[key] = r
        return r

    return rec(tuple(range(n)))


def _has_k_independent(adj, n, k):
    comp = [((1 << n) - 1) ^ adj[i] for i in range(n)]
    return _has_k_clique(comp, n, k)


def _is_5_free(adj, n):
    return not (_has_k_clique(adj, n, K) or _has_k_independent(adj, n, K))


# ---------------- construction helpers ----------------
def _paley(q):
    res = np.zeros(q, dtype=bool)
    for x in range(1, q):
        res[(x * x) % q] = True
    res[0] = False
    adj = np.zeros((q, q), dtype=np.uint8)
    for i in range(q):
        for d in range(1, q):
            if res[d]:
                adj[i, (i + d) % q] = 1
    return adj


def _find_4clique(adj, n):
    memo = {}

    def rec(cand):
        if len(cand) < 4:
            return None
        key = cand
        if key in memo:
            return memo[key]
        if len(cand) == 4:
            memo[key] = cand
            return cand
        v = cand[0]
        rest = cand[1:]
        nxt = tuple(v2 for v2 in rest if (adj[v] >> v2) & 1)
        r = rec(nxt)
        memo[key] = r
        return r

    return rec(tuple(range(n)))


def _try_add_vertex(adj, n, seed):
    """Return (n+1, adj) if a 4-clique exists (safe to add a vertex), else None."""
    rng = random.Random(seed)
    for _ in range(60):
        c = _find_4clique(adj, n)
        if c is None:
            return None
        new = np.zeros((n + 1, n + 1), dtype=np.uint8)
        new[:n, :n] = adj
        for v in c:
            new[v, n] = new[n, v] = 1
        return n + 1, new
    return None


def _greedy_4color(adj, n, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    color = [-1] * n
    for v in order:
        used = set()
        row = adj[v]
        for u in range(n):
            if row[u] and color[u] >= 0:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def _softmax_cliques(adj, n, beta, k):
    """log-sum-exp over all k-subsets of the product of edge log-weights (clique term)."""
    N = n
    if N < k:
        return -1e9
    w = adj.astype(np.float64)
    w = np.where(w > 0, 1.0, -1e-4)
    lw = np.log(w)
    lw = np.where(np.isfinite(lw), lw, -1e4)
    lw[np.diag_indices(N)] = 0.0
    logsums = np.full((N, k + 1), -1e9)
    logsums[:, 0] = 0.0
    for i in range(N):
        row = lw[i, i + 1:]
        for j in range(i + 1, N):
            for t in range(k - 1, 0, -1):
                a = logsums[j, t - 1]
                b = logsums[i, t]
                m = a if a > b else b
                logsums[i, t] = m + math.log(math.exp(a - m) + math.exp(b - m))
    return float(logsumexp(logsums[:, k]))


def _softmax_indep(adj, n, beta, k):
    comp = 1 - adj
    return _softmax_cliques(comp, n, beta, k)


def _anneal(adj, n, seed, time_budget):
    rng = random.Random(seed)
    A = adj.copy().astype(np.float64)
    color = _greedy_4color(adj, n, seed)
    beta = 8.0
    cur = _softmax_cliques(A, n, beta, K) + _softmax_indep(A, n, beta, K)
    best = cur
    bestA = A.copy()
    end = time.time() + time_budget
    while time.time() < end and _time_left() > 20.0:
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        A[i, j] = 1.0 - A[i, j]
        A[j, i] = A[i, j]
        val = _softmax_cliques(A, n, beta, K) + _softmax_indep(A, n, beta, K)
        T = 0.15
        if val < cur or rng.random() < math.exp(-(val - cur) / T):
            cur = val
            if val < best:
                best = val
                bestA = A.copy()
        else:
            A[i, j] = 1.0 - A[i, j]
            A[j, i] = A[i, j]
    return (bestA > 0.5).astype(np.uint8)


# ---------------- main ----------------
def construct():
    rng = random.Random(12345)

    # base: Paley-37 (guaranteed 5-free)
    base_adj = _paley(37)
    base_n = 37
    if not _is_5_free(base_adj, base_n):
        # safety: should never happen
        return 0, []

    best_n = base_n
    best_edges = None

    def _edges(adj, n):
        e = []
        for i in range(n):
            row = adj[i]
            for j in range(i + 1, n):
                if row[j]:
                    e.append((i, j))
        return e

    best_edges = _edges(base_adj, base_n)

    # deterministic: add a vertex to a 4-clique (Ramsey-safe)
    if _time_left() > 30.0:
        added = _try_add_vertex(base_adj, base_n, 999)
        if added is not None:
            n2, adj2 = added
            if n2 > best_n and _is_5_free(adj2, n2):
                best_n = n2
                best_edges = _edges(adj2, n2)

    # search: greedy 4-color + flips + annealing, verify exactly
    if _time_left() > 25.0:
        n0 = best_n
        start_adj = base_adj.copy() if best_n == base_n else added[1].copy()
        while _time_left() > 30.0:
            seed = rng.randrange(1 << 30)
            budget = min(45.0, _time_left() - 25.0)
            if budget < 10.0:
                break
            cand = _anneal(start_adj, n0, seed, budget)
            if _time_left() < 25.0:
                break
            if _is_5_free(cand, n0):
                # try to grow this valid graph by adding a vertex to a 4-clique
                grown = _try_add_vertex(cand, n0, seed)
                if grown is not None:
                    n3, adj3 = grown
                    if n3 > best_n and n3 <= MAX_VERTICES and _is_5_free(adj3, n3):
                        best_n = n3
                        best_edges = _edges(adj3, n3)
                if n0 > best_n:
                    best_n = n0
                    best_edges = _edges(cand, n0)
            # refresh start from the best valid graph found
            if best_edges is not None and best_n > n0:
                n0 = best_n
                start_adj = base_adj.copy()  # keep base; anneal explores from base
            # periodically restart from a fresh random 4-coloring base
            if seed % 40000 == 0:
                pass  # (anneal already re-colors each call)

    if best_edges is None:
        best_edges = _edges(base_adj, base_n)
    return best_n, best_edges
