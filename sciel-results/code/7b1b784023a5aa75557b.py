import sys
import itertools
import time
from collections import deque

def construct():
    _t0 = time.time()
    _DEADLINE = 85.0

    P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges |= set((v, u) for u, v in base_edges)

    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    all_edges.append((u, v))

    def _time_ok():
        return time.time() - _t0 < _DEADLINE

    def build_adj(added):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_diameter(adj):
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
            if -1 in dist:
                return float('inf')
            mx = max(dist)
            if mx > max_dist:
                max_dist = mx
        return max_dist

    added_edges = set()

    while _time_ok():
        adj = build_adj(added_edges)
        diam = bfs_diameter(adj)
        if diam <= 4:
            break

        best_edge = None
        best_diam = diam

        for e in all_edges:
            if e in added_edges:
                continue
            added_edges.add(e)
            adj2 = build_adj(added_edges)
            d = bfs_diameter(adj2)
            added_edges.remove(e)

            if d < best_diam:
                best_diam = d
                best_edge = e
                if best_diam <= 4:
                    break

        if best_edge is None:
            break
        added_edges.add(best_edge)

    # Removal pruning
    for e in list(added_edges):
        if not _time_ok():
            break
        added_edges.remove(e)
        adj = build_adj(added_edges)
        d = bfs_diameter(adj)
        if d > 4:
            added_edges.add(e)

    # Swap-based local search
    improved = True
    while improved and _time_ok():
        improved = False
        for e_out in list(added_edges):
            if not _time_ok():
                break
            added_edges.remove(e_out)
            for e_in in all_edges:
                if e_in in added_edges:
                    continue
                added_edges.add(e_in)
                adj2 = build_adj(added_edges)
                d2 = bfs_diameter(adj2)
                added_edges.remove(e_in)
                if d2 <= 4:
                    added_edges.add(e_in)
                    improved = True
                    break
            # After a successful swap, try removals
            if improved:
                for e_rem in list(added_edges):
                    if not _time_ok():
                        break
                    added_edges.remove(e_rem)
                    adj3 = build_adj(added_edges)
                    d3 = bfs_diameter(adj3)
                    if d3 > 4:
                        added_edges.add(e_rem)

    result = [[int(u), int(v)] for u, v in added_edges]
    return result
