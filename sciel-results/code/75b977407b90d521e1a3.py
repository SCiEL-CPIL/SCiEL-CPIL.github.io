import time
import random
import numpy as np

N = 40
DEADLINE = time.time() + 110.0  # safety margin before the 120s kill
RNG = random.Random(12345)


def _has_c6(adj):
    # adj: list of int bitmasks. C6 exists iff some pair shares >=3 common neighbors.
    for i in range(N):
        ai = adj[i]
        for j in range(i + 1, N):
            if (ai & adj[j]).bit_count() >= 3:
                return True
    return False


def _edge_count(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2


def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _seed_adj():
    # Two K10 cliques (guaranteed C6-free) + random cross edges.
    adj = [0] * N
    for a in range(10):
        for b in range(a + 1, 10):
            _add_edge(adj, a, b)
    for a in range(10, 20):
        for b in range(a + 1, 20):
            _add_edge(adj, a, b)
    for a in range(10):
        for b in range(10, 20):
            if RNG.random() < 0.35:
                _add_edge(adj, a, b)
    return adj


def _greedy_add(adj):
    # Randomized greedy: add edges that keep the graph C6-free.
    order = list(range(N * (N - 1) // 2))
    RNG.shuffle(order)
    for idx in order:
        u = idx // (N - 1)
        rem = idx % (N - 1)
        v = rem if rem < u else rem + 1
        if (adj[u] >> v) & 1:
            continue
        # quick local C6 check before commit
        au = adj[u]
        av = adj[v]
        new_au = au | (1 << v)
        new_av = av | (1 << u)
        ok = True
        for w in range(N):
            if w == u or w == v:
                continue
            if (new_au & adj[w]).bit_count() >= 3 or (new_av & adj[w]).bit_count() >= 3:
                ok = False
                break
        if ok:
            _add_edge(adj, u, v)
    return adj


def _local_search(adj):
    # Add edges (greedy) + 2-opt flips (remove e1, add e2) to maximize edges.
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        # pass 1: add
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    continue
                au = adj[u]
                av = adj[v]
                new_au = au | (1 << v)
                new_av = av | (1 << u)
                ok = True
                for w in range(N):
                    if w == u or w == v:
                        continue
                    if (new_au & adj[w]).bit_count() >= 3 or (new_av & adj[w]).bit_count() >= 3:
                        ok = False
                        break
                if ok:
                    _add_edge(adj, u, v)
                    improved = True
        # pass 2: 2-opt flips
        edges = []
        for u in range(N):
            m = adj[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                m -= lsb
        RNG.shuffle(edges)
        for (e1u, e1v) in edges:
            if time.time() > DEADLINE - 5:
                break
            if not ((adj[e1u] >> e1v) & 1):
                continue  # edge no longer present
            # candidate new edges (not present, not e1)
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) == (e1u, e1v):
                        continue
                    if (adj[u] >> v) & 1:
                        continue
                    # simulate remove e1 + add (u,v)
                    saved = adj[:]
                    _remove_edge(adj, e1u, e1v)
                    _add_edge(adj, u, v)
                    if not _has_c6(adj):
                        improved = True
                        break
                    else:
                        adj[:] = saved
                if improved:
                    break
            if improved:
                break
    return adj


def _to_edges(adj):
    out = []
    for u in range(N):
        m = adj[u]
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if v > u:
                out.append([u, v])
            m -= lsb
    return out


def construct():
    best_adj = None
    best_count = -1

    # Build a solid initial graph
    adj = _seed_adj()
    adj = _greedy_add(adj)
    adj = _local_search(adj)
    c = _edge_count(adj)
    if c > best_count:
        best_count = c
        best_adj = adj[:]

    # Random restarts to explore the space
    attempts = 0
    while time.time() < DEADLINE - 20:
        attempts += 1
        adj = _seed_adj()
        adj = _greedy_add(adj)
        adj = _local_search(adj)
        c = _edge_count(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
        if best_count >= 105:
            break

    # Final polish on the best graph with remaining time
    if time.time() < DEADLINE - 10 and best_adj is not None:
        adj = best_adj[:]
        adj = _local_search(adj)
        c = _edge_count(adj)
        if c > best_count:
            best_count = c
            best_adj = adj

    return _to_edges(best_adj)
