"""Path decomposition via maximum matching in the line graph (bipartite double cover)."""
# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - 15.0 - (time.time() - _START)

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hopcroft_karp(adj, n_left, n_right):
    from collections import deque
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        q = deque()
        found = False
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return pair_u, pair_v, matching

def _decompose(edges):
    m = len(edges)
    if m == 0:
        return []
    adj_edges = [[] for _ in range(m)]
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if i != j and (a == c or a == d or b == c or b == d):
                adj_edges[i].append(j)
    if _remaining() < 5.0:
        return [[list(e)] for e in edges]
    pair_u, pair_v, _ = _hopcroft_karp(adj_edges, m, m)
    succ = [pair_u[i] if pair_u[i] != -1 else -1 for i in range(m)]
    pred = [-1] * m
    for j in range(m):
        if pair_v[j] != -1:
            pred[j] = pair_v[j]
    visited = [False] * m
    paths = []
    def shared_vertex(e1, e2):
        a, b = e1
        c, d = e2
        if a == c or a == d:
            return a
        if b == c or b == d:
            return b
        return None
    for start in range(m):
        if visited[start]:
            continue
        cur = start
        while pred[cur] != -1 and not visited[pred[cur]]:
            cur = pred[cur]
        path_edges = []
        node = cur
        seen = set()
        while node != -1 and not visited[node]:
            if node in seen:
                break
            seen.add(node)
            visited[node] = True
            path_edges.append(node)
            node = succ[node]
        if not path_edges:
            continue
        verts = [list(edges[path_edges[0]])]
        for k in range(1, len(path_edges)):
            sv = shared_vertex(edges[path_edges[k - 1]], edges[path_edges[k]])
            a, b = edges[path_edges[k]]
            verts.append(b if a == sv else a)
        paths.append(verts)
    for i in range(m):
        if not visited[i]:
            paths.append([list(edges[i])])
    return paths

def _verify(paths, edges):
    eset = set(edges)
    used = set()
    for p in paths:
        if len(p) < 1:
            return False
        if len(set(p)) != len(p):
            return False
        for i in range(len(p) - 1):
            e = tuple(sorted((p[i], p[i + 1])))
            if e not in eset or e in used:
                return False
            used.add(e)
    return used == eset

def construct():
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(*spec)
        paths = _decompose(edges)
        if not _verify(paths, edges):
            paths = [[list(e)] for e in edges]
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
