"""Decompose three fixed instance graphs into edge-disjoint cycles + single edges, minimizing pieces."""
import time
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_long_cycle(adj, n, start, rng, node_cap):
    """Iterative DFS to find a cycle through `start`. Returns list of vertices (no repeat of start) or None."""
    stack = [(start, None, None, 0)]  # (node, parent, prev_iter_idx, depth)
    # We do a DFS that records parent pointers; when we hit a back edge to an ancestor, extract cycle.
    parent = [-1] * n
    depth = [-1] * n
    visited = [False] * n
    order_stack = [start]
    parent[start] = -2
    depth[start] = 0
    visited[start] = True
    iter_stack = [list(adj[start])]
    it_idx = [0]
    steps = 0
    max_steps = node_cap
    while order_stack:
        if steps > max_steps:
            return None
        steps += 1
        node = order_stack[-1]
        nbrs = iter_stack[-1]
        idx = it_idx[-1]
        # find next unvisited neighbor
        found = False
        while idx < len(nbrs):
            nb = nbrs[idx]
            idx += 1
            if not visited[nb]:
                it_idx[-1] = idx
                visited[nb] = True
                parent[nb] = node
                depth[nb] = depth[node] + 1
                order_stack.append(nb)
                iter_stack.append(list(adj[nb]))
                it_idx.append(0)
                found = True
                break
            else:
                # back edge: check if nb is a proper ancestor (depth smaller and nb != parent)
                if nb != parent[node] and depth[nb] < depth[node]:
                    # cycle from nb .. node
                    cyc = [node]
                    cur = node
                    while cur != nb:
                        cur = parent[cur]
                        cyc.append(cur)
                    # cyc is node..nb; reverse so it starts at nb
                    cyc = cyc[::-1]
                    return cyc
        if not found:
            order_stack.pop()
            iter_stack.pop()
            it_idx.pop()
    return None


def _cycle_edges(cyc):
    """Given cycle vertex list (distinct, length>=3), return set of edges (closed)."""
    k = len(cyc)
    es = set()
    for i in range(k):
        a, b = cyc[i], cyc[(i + 1) % k]
        es.add((a, b) if a < b else (b, a))
    return es


def _decompose(edges, n, deadline, rng, node_cap):
    remaining = set(edges)
    adj = _adjacency(edges, n)
    pieces = []
    start_times = []
    while remaining:
        if time.time() >= deadline:
            break
        # pick a start vertex with highest degree in remaining
        deg = [len(adj[v]) for v in range(n)]
        # choose a vertex with degree>=2 that has the most remaining incident edges
        best_v = -1
        best_d = 0
        for v in range(n):
            if deg[v] >= 2 and deg[v] > best_d:
                best_d = deg[v]
                best_v = v
        if best_v < 0:
            break
        # try a few random starts to find a cycle
        found = None
        attempts = 0
        max_attempts = 8
        # first try best_v, then random high-degree vertices
        candidates = [best_v]
        for _ in range(7):
            cand = rng.choice([v for v in range(n) if deg[v] >= 2])
            if cand not in candidates:
                candidates.append(cand)
        for s in candidates:
            attempts += 1
            cyc = _find_long_cycle(adj, n, s, rng, node_cap)
            if cyc and len(cyc) >= 3:
                found = cyc
                break
            if time.time() >= deadline:
                break
        if found:
            cyc_edges = _cycle_edges(found)
            # verify all present
            if cyc_edges <= remaining:
                pieces.append(found)
                for e in cyc_edges:
                    remaining.discard(e)
                    a, b = e
                    adj[a].discard(b)
                    adj[b].discard(a)
                continue
        # no cycle found: peel a single edge
        # pick an edge whose removal is "least disruptive": choose edge with min max-degree endpoints
        # simply pick first edge
        e = next(iter(remaining))
        remaining.discard(e)
        a, b = e
        adj[a].discard(b)
        adj[b].discard(a)
        pieces.append([a, b])
    # After loop, any remaining edges (due to deadline) become singletons
    for e in remaining:
        pieces.append(list(e))
    return pieces


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill
    rng = Random(12345)
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for (n, p, seed) in specs:
        edges = _instance(n, p, seed)
        if time.time() >= deadline:
            # not enough time; return all edges as singletons
            results.append([list(e) for e in edges])
            continue
        node_cap = 6000
        pieces = _decompose(edges, n, deadline, rng, node_cap)
        results.append(pieces)
    return results
