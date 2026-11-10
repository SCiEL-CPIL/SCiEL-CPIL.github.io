"""Minimize four-cycles in a 31-vertex, 91-edge graph."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

N = 31
E = 91
DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _polarity_edges():
    """Edges of the polarity graph of the unitals of PG(2,5).

    Vertex i (0..30) is point (1, x, y) with x = i // 6, y = i % 6.
    Point P is incident with line L if  X*Y + Y*Z + Z*X == 0  (over GF(5)).
    The polarity graph is 6-regular and C4-free (90 edges).
    """
    def eval_form(X, Y, Z):
        return (X * Y + Y * Z + Z * X) % 5

    edges = []
    for i in range(N):
        xi, yi = i // 6, i % 6
        Pi = (1, xi, yi)
        for j in range(i + 1, N):
            xj, yj = j // 6, j % 6
            Pj = (1, xj, yj)
            # incidence: Pj on line Pi  <=>  eval_form(Pi, Pj) == 0
            if eval_form(Pi[0], Pi[1], Pi[2], ) if False else eval_form(Pi[0], Pi[1], Pi[2]) if False else (Pi[0]*Pj[1] + Pi[1]*Pj[2] + Pi[2]*Pj[0]) % 5 == 0:
                edges.append((i, j))
    return edges


def _build_index(edges):
    """Return (edge_index, neighbors, adj, common)."""
    idx = {}
    neigh = [set() for _ in range(N)]
    for a, b in edges:
        idx[(a, b)] = len(idx)
        idx[(b, a)] = len(idx)
        neigh[a].add(b)
        neigh[b].add(a)
    adj = np.zeros((N, N), dtype=bool)
    for a in range(N):
        for b in neigh[a]:
            adj[a, b] = True
    common = np.zeros((N, N), dtype=np.int64)
    for a in range(N):
        for b in range(N):
            common[a, b] = int(np.sum(adj[a] & adj[b]))
    return idx, neigh, adj, common


def _count_c4(edges):
    """Exact number of 4-cycles (subgraphs)."""
    idx, neigh, adj, common = _build_index(edges)
    cnt = 0
    for a in range(N):
        for b in range(a + 1, N):
            cnt += common[a, b]
    return cnt // 2


def _delta_c4(common, e):
    """Number of new 4-cycles created by adding edge e to the current graph.

    A new 4-cycle uses e as one of its two diagonals, so it is counted exactly
    once as common[u][v] for the endpoints (u, v) of e.
    """
    return int(common[e[0]][e[1]])


def _add(common, adj, neigh, e):
    u, v = e
    neigh[u].add(v)
    neigh[v].add(u)
    adj[u, v] = adj[v, u] = True
    for x in neigh[u]:
        common[x, v] += 1
        common[v, x] += 1
    for x in neigh[v]:
        common[x, u] += 1
        common[u, x] += 1


def _remove(common, adj, neigh, e):
    u, v = e
    for x in neigh[u]:
        if x != v:
            common[x, v] -= 1
            common[v, x] -= 1
    for x in neigh[v]:
        if x != u:
            common[x, u] -= 1
            common[u, x] -= 1
    neigh[u].discard(v)
    neigh[v].discard(u)
    adj[u, v] = adj[v, u] = False


def _greedy_addition(base_edges, pool, common, adj, neigh, k):
    """Greedily add k edges from pool, always taking the smallest delta."""
    chosen = []
    for _ in range(k):
        best = None
        best_d = None
        for e in pool:
            d = _delta_c4(common, e)
            if best_d is None or d < best_d:
                best_d = d
                best = e
        _add(common, adj, neigh, best)
        chosen.append(best)
    return chosen


def _sa(base_edges, pool, k, start, budget, seed):
    """Simulated annealing over the set of k added edges (order-independent)."""
    rng = random.Random(seed)
    base_set = set(map(frozenset, base_edges))
    pool_set = [frozenset(e) for e in pool]
    pool_by_edge = {tuple(sorted(e)): e for e in pool}

    def edge_from_fs(fs):
        a, b = tuple(sorted(fs))
        return (a, b)

    # initial state: greedy pick
    common, adj, neigh = _build_index(base_edges)
    cur = _greedy_addition(base_edges, pool, common, adj, neigh, k)
    cur_fs = set(frozenset(e) for e in cur)
    cur_c4 = _count_c4(base_edges + cur)

    best = list(cur)
    best_c4 = cur_c4

    T = 12.0
    T_end = 0.05
    t0 = time.time()
    it = 0
    while True:
        if time.time() - t0 > budget:
            break
        it += 1
        # propose: swap one added edge for one pool edge
        a = rng.randrange(k)
        removed = cur[a]
        _remove(common, adj, neigh, removed)
        cur_fs.discard(frozenset(removed))

        candidates = [fs for fs in pool_set if fs not in cur_fs]
        if not candidates:
            _add(common, adj, neigh, removed)
            cur_fs.add(frozenset(removed))
            continue
        new_fs = rng.choice(candidates)
        new_e = edge_from_fs(new_fs)
        _add(common, adj, neigh, new_e)
        cur_fs.add(new_fs)
        new_c4 = _count_c4(base_edges + [edge_from_fs(x) for x in cur_fs])

        d = new_c4 - cur_c4
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur_c4 = new_c4
            cur[a] = new_e
            if new_c4 < best_c4:
                best_c4 = new_c4
                best = [edge_from_fs(x) for x in cur_fs]
        else:
            # revert
            _remove(common, adj, neigh, new_e)
            cur_fs.discard(new_fs)
            _add(common, adj, neigh, removed)
            cur_fs.add(frozenset(removed))

        if it % 200 == 0:
            T = max(T_end, T * 0.985)

    return best, best_c4


def construct():
    t_start = time.time()

    base_edges = _polarity_edges()
    base_set = set(map(frozenset, base_edges))
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pool = [e for e in all_edges if frozenset(e) not in base_set]
    k = E - len(base_edges)  # 1

    # fast exact baseline
    common, adj, neigh = _build_index(base_edges)
    greedy = _greedy_addition(base_edges, pool, common, adj, neigh, k)
    best_edges = base_edges + greedy
    best_c4 = _count_c4(best_edges)

    # polish with simulated annealing (cheap: k == 1)
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 5:
        sa_edges, sa_c4 = _sa(base_edges, pool, k, 0, remaining - 6, seed=12345)
        if sa_c4 < best_c4:
            best_c4 = sa_c4
            best_edges = base_edges + sa_edges

    # final verification (must be exactly 91 distinct edges on 0..30)
    seen = set()
    for a, b in best_edges:
        assert 0 <= a < N and 0 <= b < N and a != b
        key = (min(a, b), max(a, b))
        assert key not in seen
        seen.add(key)
    assert len(best_edges) == E

    return [[a, b] for a, b in best_edges]
# EVOLVE-BLOCK-END
