"""Rewritten program: greedy vertex-disjoint cycle extraction for edge decomposition."""
# EVOLVE-BLOCK-START
import time
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    n = max((max(e) for e in edges), default=0) + 1
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    used_edges = set()
    visited = [False] * n

    def find_cycle():
        # iterative DFS to find any cycle among unvisited vertices with free edges
        for s in range(n):
            if visited[s]:
                continue
            if not adj[s]:
                visited[s] = True
                continue
            # BFS/DFS with parent tracking
            parent = {s: None}
            stack = [s]
            while stack:
                u = stack.pop()
                for v in sorted(adj[u]):
                    if (min(u, v), max(u, v)) in used_edges:
                        continue
                    if v not in parent:
                        parent[v] = u
                        stack.append(v)
                    elif v != parent.get(u, None) and u != parent.get(v, None):
                        # found a cycle: u - v (back edge) plus path v..u
                        # reconstruct path from v up to u via parent
                        path = [v]
                        cur = v
                        while cur != u:
                            cur = parent[cur]
                            path.append(cur)
                        path.append(u)
                        return path
            # mark all reachable as visited (no cycle found from s)
            for x in parent:
                visited[x] = True
        return None

    while time.time() < deadline:
        cyc = find_cycle()
        if cyc is None:
            break
        # mark cycle edges used and vertices visited
        m = len(cyc)
        for i in range(m):
            a, b = cyc[i], cyc[(i + 1) % m]
            used_edges.add((min(a, b), max(a, b)))
        for x in cyc:
            visited[x] = True
        pieces.append(cyc)

    # cover remaining edges
    for u, v in edges:
        if (u, v) not in used_edges:
            pieces.append([u, v])
            used_edges.add((u, v))
    return pieces


def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # fallback: all edges as pieces
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        results.append(_decompose(edges, deadline))
    return results
# EVOLVE-BLOCK-END
