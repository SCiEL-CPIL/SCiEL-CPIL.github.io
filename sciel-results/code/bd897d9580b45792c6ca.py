"""Simulated annealing on full edge set with fast bipartition-based Q3 counter."""
import time
import random
from itertools import combinations

_Q3_PAIRS = None

def _build_q3_pairs():
    global _Q3_PAIRS
    if _Q3_PAIRS is not None:
        return _Q3_PAIRS
    all4 = list(combinations(range(16), 4))
    pairs = []
    n = len(all4)
    for i in range(n):
        Ai = set(all4[i])
        for j in range(i + 1, n):
            if not (Ai & set(all4[j])):
                pairs.append((all4[i], all4[j]))
    _Q3_PAIRS = pairs
    return pairs

def _count_q3(adj):
    pairs = _build_q3_pairs()
    cnt = 0
    for A, B in pairs:
        ok = True
        for k in range(4):
            for l in range(k + 1, 4):
                if (adj[A[k]] >> A[l]) & 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for k in range(4):
            for l in range(k + 1, 4):
                if (adj[B[k]] >> B[l]) & 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        Bmask = 0
        for v in B:
            Bmask |= (1 << v)
        cross = 0
        for u in A:
            cross += (adj[u] & Bmask).bit_count()
        if cross != 12:
            continue
        ok = True
        missing = []
        for u in A:
            non = 0
            for v in B:
                if not ((adj[u] >> v) & 1):
                    non |= (1 << v)
            if non.bit_count() != 1:
                ok = False
                break
            missing.append(non)
        if not ok:
            continue
        if len(set(missing)) != 4:
            continue
        cnt += 1
    return cnt

_ALL_EDGES = [(u, v) for u in range(16) for v in range(u + 1, 16)]

def _build_adj(edges_set):
    adj = [0] * 16
    for (u, v) in edges_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_q3_free(edges_set):
    adj = _build_adj(edges_set)
    return _count_q3(adj) == 0

def construct():
    start = time.time()
    DEADLINE = start + 140.0
    random.seed(777)
    _build_q3_pairs()

    edges = set()
    for (u, v) in _ALL_EDGES:
        if random.random() < 0.75:
            edges.add((u, v))
    adj = _build_adj(edges)

    cur_count = _count_q3(adj)
    cur_len = len(edges)

    best_edges = set(edges) if cur_count == 0 else None
    best_len = cur_len if cur_count == 0 else 0

    T = 1.5
    T_min = 0.05
    M = 500.0
    cur_E = -cur_len + M * cur_count
    import math

    while time.time() < DEADLINE - 3.0:
        u, v = random.choice(_ALL_EDGES)
        present = (u, v) in edges
        if present:
            edges.discard((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        new_count = _count_q3(adj)
        new_len = len(edges)
        new_E = -new_len + M * new_count

        dE = new_E - cur_E
        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-9)):
            cur_count = new_count
            cur_len = new_len
            cur_E = new_E
            if cur_count == 0 and cur_len > best_len:
                best_len = cur_len
                best_edges = set(edges)
        else:
            if present:
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                edges.discard((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

        T = max(T_min, T * 0.9995)

    if best_edges is not None and _is_q3_free(best_edges):
        return [[u, v] for (u, v) in best_edges]
    # fallback K3,13: 39 edges, Q3-free
    return [[a, b] for a in range(3) for b in range(3, 16)]
