"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
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


def _build_adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def _decompose(edges, rng):
    """Greedy edge-disjoint path cover. Returns list of paths (vertex lists)."""
    n = max(max(e) for e in edges) + 1 if edges else 0
    # multigraph edge tracking: count of unused copies of each undirected edge
    remaining = {}
    for u, v in edges:
        key = (u, v) if u < v else (v, u)
        remaining[key] = remaining.get(key, 0) + 1
    # adjacency with multiplicity
    adj = {i: [] for i in range(n)}
    for (u, v), c in remaining.items():
        for _ in range(c):
            adj[u].append(v)
            adj[v].append(u)
    # shuffle adjacency for randomization
    for i in range(n):
        rng.shuffle(adj[i])

    # odd-degree vertices (by total degree in the multigraph)
    deg = {i: len(adj[i]) for i in range(n)}
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    rng.shuffle(odd)

    # pointer into each adjacency list (only pop from the "front" we manage)
    # We'll maintain per-vertex list and index; simpler: use list and pop(0) is slow.
    # Use a dict of deques via list + index pointer.
    idx = {i: 0 for i in range(n)}

    def neighbors_of(v):
        L = adj[v]
        i = idx[v]
        while i < len(L) and remaining.get((min(v, L[i]), max(v, L[i])), 0) == 0:
            i += 1
        idx[v] = i
        if i < len(L):
            return L[i]
        return None

    def use_edge(a, b):
        key = (a, b) if a < b else (b, a)
        remaining[key] -= 1

    paths = []
    # start points: odd vertices first, then any vertex with remaining edges
    starts = list(odd)
    for i in range(n):
        if i not in odd:
            starts.append(i)
    rng.shuffle(starts)

    used_start = set()
    for s in starts:
        if s in used_start:
            continue
        # does s still have an unused incident edge?
        if neighbors_of(s) is None:
            continue
        used_start.add(s)
        # build a path starting at s
        path = [s]
        cur = s
        while True:
            nb = neighbors_of(cur)
            if nb is None:
                break
            use_edge(cur, nb)
            path.append(nb)
            cur = nb
        paths.append(path)
    return paths


def _verify(paths, edges):
    """Check simple, uses edges, edge-disjoint, covers all edges."""
    seen = set()
    for p in paths:
        # simple
        if len(set(p)) != len(p):
            return False
        for a, b in zip(p, p[1:]):
            key = (a, b) if a < b else (b, a)
            if key in seen:
                return False
            seen.add(key)
    return seen == set((u, v) if u < v else (v, u) for u, v in edges)


def _solve_one(edges, deadline, base_seed):
    rng0 = Random(base_seed)
    # first greedy pass
    best = _decompose(edges, rng0)
    best_cnt = len(best)
    if not _verify(best, edges):
        # fallback: each edge its own path
        best = [[u, v] for u, v in edges]
        best_cnt = len(best)
    # randomized restarts
    rng = Random(base_seed + 777)
    t0 = time.time()
    it = 0
    while time.time() < deadline:
        it += 1
        cand = _decompose(edges, rng)
        if not _verify(cand, edges):
            continue
        if len(cand) < best_cnt:
            best = cand
            best_cnt = len(cand)
            if best_cnt == 1:
                break
    return best


def construct():
    deadline = time.time() + 140  # safety margin before 150s wall-clock
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    per = (deadline - time.time()) / 3.0
    results = []
    for k, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        d = deadline - (len(specs) - k - 1) * per
        results.append(_solve_one(edges, d, seed))
    return results
# EVOLVE-BLOCK-END
