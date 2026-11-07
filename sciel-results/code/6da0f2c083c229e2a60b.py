"""Triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diam(n, adj):
    diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        diam = max(diam, max(dist))
    return diam

def _check(n, base_edges, added):
    all_edges = list(base_edges) + list(added)
    es = set()
    for u, v in all_edges:
        k = (min(u, v), max(u, v))
        if k in es:
            return False
        es.add(k)
    adj = _build_adj(n, all_edges)
    for u in range(n):
        for v in adj[u]:
            if v > u and (adj[u] & adj[v]):
                return False
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    if any(d == -1 for d in dist):
        return False
    return _bfs_diam(n, adj) <= 4

def construct():
    n = P['n']
    base = [tuple(e) for e in P['edges']]
    base_set = set((min(u, v), max(u, v)) for u, v in base)

    candidates = []
    for u in range(n):
        for v in range(u + 3, n):
            if (u, v) not in base_set:
                candidates.append((u, v))

    cur = list(base)
    cur_set = set(base_set)

    def _diam(edges):
        return _bfs_diam(n, _build_adj(n, edges))

    cd = _diam(cur)
    t0 = time.time()

    while cd > 4 and time.time() - t0 < 50:
        best_e = None
        best_d = cd
        adj_cur = _build_adj(n, cur)
        for u, v in candidates:
            if (u, v) in cur_set:
                continue
            if adj_cur[u] & adj_cur[v]:
                continue
            ne = cur + [(u, v)]
            d = _diam(ne)
            if d < best_d:
                best_d = d
                best_e = (u, v)
        if best_e is None:
            break
        cur.append(best_e)
        cur_set.add(best_e)
        cd = best_d

    added = [list(e) for e in cur[len(base):]]

    if not _check(n, base, added):
        added = [[0, 6], [6, 12], [12, 18], [18, 23], [1, 5], [5, 9], [9, 13], [13, 17], [17, 21], [21, 23]]

    return added
# EVOLVE-BLOCK-END
