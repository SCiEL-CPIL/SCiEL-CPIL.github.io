import time
import numpy as np

N = 40
M = 401
_ALL = [(i, j) for i in range(N) for j in range(i + 1, N)]


def _books(adj, edges):
    best = 0
    books = {}
    for (u, v) in edges:
        c = len(adj[u] & adj[v])
        books[(u, v)] = c
        if c > best:
            best = c
    return books, best


def _apply_swap(adj, edges, rem, add):
    new_edges = [e for e in edges if e != rem] + [add]
    new_adj = [set(s) for s in adj]
    a, b = rem
    new_adj[a].discard(b)
    new_adj[b].discard(a)
    c, d = add
    new_adj[c].add(d)
    new_adj[d].add(c)
    return new_edges, new_adj


def _local_search(edges, adj, deadline, rng):
    books, cur = _books(adj, edges)
    while time.time() < deadline:
        if cur <= 7:
            break
        worst = max(books, key=lambda e: books[e])
        u, v = worst
        common = adj[u] & adj[v]
        present = set(edges)
        absent = [e for e in _ALL if e not in present]
        if not absent:
            break
        best_move = None
        best_score = cur
        rem_set = []
        for w in common:
            rem_set.append((u, w))
        rem_set = [e for e in set(rem_set) if e in present]
        for rem in rem_set:
            add = absent[rng.integers(len(absent))]
            if add == rem:
                continue
            ne, na = _apply_swap(adj, edges, rem, add)
            _, ns = _books(na, ne)
            if ns < best_score:
                best_score = ns
                best_move = (rem, add, ne, na)
        if best_move is not None:
            rem, add, ne, na = best_move
            edges, adj = ne, na
            books, cur = _books(adj, edges)
            continue
        # escape local minimum with a non-worsening random swap
        moved = False
        for _ in range(300):
            if time.time() >= deadline:
                break
            rem = edges[rng.integers(len(edges))]
            add = absent[rng.integers(len(absent))]
            if add == rem:
                continue
            ne, na = _apply_swap(adj, edges, rem, add)
            _, ns = _books(na, ne)
            if ns <= cur:
                edges, adj = ne, na
                books, cur = _books(adj, edges)
                moved = True
                break
        if not moved:
            break
    return edges, adj, cur


def _random_start(rng):
    idx = rng.choice(len(_ALL), size=M, replace=False)
    edges = [_ALL[i] for i in idx]
    adj = [set() for _ in range(N)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return edges, adj


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under the 150s wall-clock kill
    rng = np.random.default_rng(20240601)
    best_edges = None
    best_score = 10**9

    edges, adj = _random_start(rng)
    while time.time() < deadline:
        edges, adj, cur = _local_search(edges, adj, deadline, rng)
        if cur < best_score:
            best_score = cur
            best_edges = list(edges)
        if best_score <= 7:
            break
        edges, adj = _random_start(rng)

    if best_edges is None:
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    out = []
    seen = set()
    for (u, v) in best_edges:
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1]])
    if len(out) < M:
        for e in _ALL:
            if len(out) >= M:
                break
            if e not in seen:
                seen.add(e)
                out.append([e[0], e[1]])
    return out[:M]
