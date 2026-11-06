"""Direct constructive + local-improvement path decomposition."""
import sys
import time
from collections import defaultdict, deque

# ---- deadline guard -------------------------------------------------------
_START = time.monotonic()
_LIMIT = 140.0   # safety margin under the 150s kill


def _remaining():
    return _LIMIT - (time.monotonic() - _START)


# ---- graph construction (copied from verifier) ----------------------------
def _build_graph(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


# ---- graph helpers --------------------------------------------------------
def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _valid_decomposition(paths, edge_set):
    seen = set()
    for p in paths:
        if len(p) < 2:
            return False
        if len(set(p)) != len(p):
            return False
        for i in range(len(p) - 1):
            e = tuple(sorted((p[i], p[i + 1])))
            if e not in edge_set or e in seen:
                return False
            seen.add(e)
    return seen == edge_set


def _edge_seq_of(path):
    return [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]


def _try_improve(paths, edge_set):
    """Greedy: repeatedly merge any two paths that can combine into a simple path."""
    improved = True
    while improved and _remaining() > 1.0:
        improved = False
        n = len(paths)
        for i in range(n):
            if _remaining() < 0.5:
                break
            for j in range(i + 1, n):
                a, b = paths[i], paths[j]
                # Try all four endpoint combinations
                merged = None
                # a_end + b_start
                if a[-1] == b[0]:
                    cand = a + b[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                # a_start + b_end
                if merged is None and a[0] == b[-1]:
                    cand = b + a[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                # b_end + a_start
                if merged is None and b[-1] == a[0]:
                    cand = b + a[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand
                # b_start + a_end
                if merged is None and b[0] == a[-1]:
                    cand = b + a[1:]
                    if len(set(cand)) == len(cand):
                        merged = cand

                if merged is not None:
                    paths = paths[:i] + paths[i + 1:j] + paths[j + 1:] + [merged]
                    improved = True
                    break
            if improved:
                break
    return paths


# ---- constructive decomposition ------------------------------------------
def _constructive(adj, edges):
    """Return (paths, remaining_edges)."""
    paths = []
    rem = set(tuple(sorted(e)) for e in edges)

    # 1) Peel off cycles (even subgraphs) to reduce odd-degree vertices
    while True:
        # Find a vertex with degree >= 2 in remaining graph
        deg = defaultdict(int)
        for u, v in rem:
            deg[u] += 1
            deg[v] += 1

        start = None
        for v in deg:
            if deg[v] >= 2:
                start = v
                break
        if start is None:
            break

        # Walk to find a cycle
        cycle = [start]
        prev = None
        cur = start
        ok = False
        guard = 0
        while guard < 10000:
            guard += 1
            nbrs = [w for w in adj[cur] if tuple(sorted((cur, w))) in rem]
            nxt = None
            for w in nbrs:
                if w != prev:
                    nxt = w
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            if cur == start:
                ok = True
                break
            cycle.append(cur)

        if not ok or len(cycle) < 3:
            break

        # Verify cycle edges are all in rem
        cycle_edges = []
        good = True
        for i in range(len(cycle) - 1):
            e = tuple(sorted((cycle[i], cycle[i + 1])))
            if e not in rem:
                good = False
                break
            cycle_edges.append(e)
        if not good:
            break

        # Remove cycle edges from rem
        for e in cycle_edges:
            rem.discard(e)
        paths.append(list(cycle))

    # 2) Pair up odd-degree vertices and find paths between them
    deg = defaultdict(int)
    for u, v in rem:
        deg[u] += 1
        deg[v] += 1

    odd = [v for v in deg if deg[v] % 2 == 1]
    i = 0
    while i + 1 < len(odd):
        u, v = odd[i], odd[i + 1]
        path = _find_path(adj, rem, u, v)
        if path is not None:
            for k in range(len(path) - 1):
                rem.discard(tuple(sorted((path[k], path[k + 1]))))
            paths.append(path)
        i += 2

    # 3) Any leftover edges -> each as its own path
    for e in list(rem):
        paths.append(list(e))
        rem.discard(e)

    return paths, rem


def _find_path(adj, rem, u, v):
    """Find a path from u to v in the remaining edge set (BFS)."""
    if u == v:
        return None
    parent = {u: None}
    q = deque([u])
    found = False
    while q:
        cur = q.popleft()
        if cur == v:
            found = True
            break
        for w in adj[cur]:
            e = tuple(sorted((cur, w)))
            if e in rem and w not in parent:
                parent[w] = cur
                q.append(w)
    if not found:
        return None
    path = [v]
    while path[-1] != u:
        path.append(parent[path[-1]])
    path.reverse()
    return path


# ---- per-instance solve ----------------------------------------------------
def _solve_instance(edges):
    adj = _build_adj(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    paths, rem = _constructive(adj, edges)
    if not _valid_decomposition(paths, edge_set):
        # fallback: each edge its own path (always valid)
        paths = [list(e) for e in edges]
    else:
        paths = _try_improve(paths, edge_set)
        if not _valid_decomposition(paths, edge_set):
            paths = [list(e) for e in edges]
    return paths


# ---- entry point -----------------------------------------------------------
def construct():
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for idx, spec in enumerate(specs):
        if _remaining() < 3.0:
            # not enough time: fill remaining with trivial decomposition
            for s in specs[len(results):]:
                edges = _build_graph(*s)
                results.append([list(e) for e in edges])
            break
        edges = _build_graph(*spec)
        results.append(_solve_instance(edges))
    return results


if __name__ == "__main__":
    out = construct()
    total = sum(len(r) for r in out)
    print("paths per instance:", [len(r) for r in out])
    print("total:", total)
