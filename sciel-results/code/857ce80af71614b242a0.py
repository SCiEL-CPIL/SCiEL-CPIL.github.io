"""Rewritten program: triangle-free diameter-four augmentation on P24 path."""
import time
import collections
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj, n):
    for u in range(n):
        nb = list(adj[u])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if nb[j] in adj[nb[i]]:
                    return False
    return True

def _diameter_bfs(adj, n):
    maxd = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if any(d == -1 for d in dist):
            return -1
        md = max(dist)
        if md > maxd:
            maxd = md
            if maxd > 4:
                return maxd
    return maxd

def _validate(added, n):
    base = set()
    for e in P['edges']:
        a, b = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        base.add((a, b))
    seen = set()
    all_edges = []
    for e in added:
        u, v = e[0], e[1]
        if not (isinstance(u, int) and not isinstance(u, bool)) or not (isinstance(v, int) and not isinstance(v, bool)):
            return False
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in base or (a, b) in seen:
            return False
        seen.add((a, b))
        all_edges.append((a, b))
    adj = _adjacency(all_edges, n)
    if not _is_triangle_free(adj, n):
        return False
    d = _diameter_bfs(adj, n)
    if d < 0 or d > 4:
        return False
    return True

def _base_set(n):
    s = set()
    for e in P['edges']:
        a, b = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        s.add((a, b))
    return s

def _design_bipartite_hubs():
    """Bipartite (parity) design -> automatically triangle-free. Add cross-parity
    edges so every vertex is within 1 of a hub and hubs are mutually close."""
    n = P['n']
    base = _base_set(n)
    even = [i for i in range(n) if i % 2 == 0]
    odd = [i for i in range(n) if i % 2 == 1]
    H_e = [4, 10, 16]
    H_o = [5, 11, 17]
    added = []
    seen = set()

    def add(u, v):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in base or (a, b) in seen:
            return
        seen.add((a, b))
        added.append([a, b])

    for v in range(n):
        if v % 2 == 0:
            h = min(H_e, key=lambda x: abs(v - x))
            add(v, h)
        else:
            h = min(H_o, key=lambda x: abs(v - x))
            add(v, h)
    for he in H_e:
        for ho in H_o:
            add(he, ho)
    return added

def _design_greedy():
    """Greedy cross-parity augmentation until diameter <= 4."""
    n = P['n']
    base = _base_set(n)
    adj = _adjacency(list(P['edges']), n)
    added = []
    added_set = set()
    even = [i for i in range(n) if i % 2 == 0]
    odd = [i for i in range(n) if i % 2 == 1]

    def diam():
        return _diameter_bfs(adj, n)

    cur = diam()
    cross = []
    for e in even:
        for o in odd:
            a, b = (e, o) if e < o else (o, e)
            if (a, b) not in base:
                cross.append((a, b))
    random.seed(1234)
    guard = 0
    while cur > 4 and guard < 800:
        guard += 1
        sample = cross if len(cross) <= 300 else random.sample(cross, 300)
        best_e = None
        best_d = cur
        for (u, v) in sample:
            if (u, v) in added_set:
                continue
            adj[u].add(v)
            adj[v].add(u)
            d = diam()
            adj[u].discard(v)
            adj[v].discard(u)
            if d <= 4:
                best_e = (u, v)
                best_d = d
                break
            if 0 < d < best_d:
                best_d = d
                best_e = (u, v)
        if best_e is None:
            break
        u, v = best_e
        adj[u].add(v)
        adj[v].add(u)
        added_set.add(best_e)
        added.append([u, v])
        cur = best_d
    return added if cur <= 4 else None

def construct():
    start = time.time()
    deadline = start + 70.0
    n = P['n']

    candidates = []
    c1 = _design_bipartite_hubs()
    if c1 is not None:
        candidates.append(c1)

    best = None
    for cand in candidates:
        if time.time() > deadline:
            break
        if _validate(cand, n):
            if best is None or len(cand) < len(best):
                best = cand

    if best is None and time.time() < deadline:
        g = _design_greedy()
        if g is not None and _validate(g, n):
            best = g

    if best is None:
        return []

    out = []
    for e in best:
        u = int(e[0])
        v = int(e[1])
        if u > v:
            u, v = v, u
        out.append([u, v])
    return out
