# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 120.0
_START = time.time()
_specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _greedy_fallback(G, n, edges):
    # Decompose into simple edge-disjoint paths via greedy longest-path.
    # Returns (paths, ok).
    used = set()
    adj = {v: [] for v in range(n)}
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    paths = []
    while len(used) < len(edges):
        # pick an unused edge with minimum degrees (to keep odd endpoints low)
        cand = None
        for (a, b) in edges:
            if (a, b) in used:
                continue
            cand = (a, b)
            break
        if cand is None:
            break
        a, b = cand
        used.add((a, b))
        path = [a, b]
        # extend forward
        while True:
            cur = path[-1]
            nxt = None
            for w in adj[cur]:
                e = (min(cur, w), max(cur, w))
                if e not in used and w not in path:
                    nxt = w
                    break
            if nxt is None:
                break
            used.add((min(cur, nxt), max(cur, nxt)))
            path.append(nxt)
        # extend backward
        while True:
            cur = path[0]
            nxt = None
            for w in adj[cur]:
                e = (min(cur, w), max(cur, w))
                if e not in used and w not in path:
                    nxt = w
                    break
            if nxt is None:
                break
            used.add((min(cur, nxt), max(cur, nxt)))
            path.insert(0, nxt)
        paths.append(path)
    ok = (len(used) == len(edges))
    return paths, ok


def _decompose(G, n, edges):
    # Returns (paths, ok).
    # Build adjacency list with edge ids.
    edge_index = {}
    for i, (a, b) in enumerate(edges):
        edge_index[(a, b)] = i
    adj = {v: [] for v in range(n)}
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i))
        adj[b].append((a, i))
    m = len(edges)
    if m == 0:
        return [], True

    # degrees
    deg = [0] * n
    for (a, b) in edges:
        deg[a] += 1
        deg[b] += 1
    odd = [v for v in range(n) if deg[v] % 2 == 1]

    # dummy edges: pair up odd vertices
    dummy = []
    if len(odd) % 2 == 1:
        # cannot happen for a valid graph; guard anyway
        return _greedy_fallback(G, n, edges)
    for i in range(0, len(odd), 2):
        u, v = odd[i], odd[i + 1]
        if u == v:
            continue
        dummy.append((u, v))

    # Build combined adjacency: real edges (id 0..m-1) + dummy edges (id m..m+len(dummy)-1)
    comb_adj = {v: [] for v in range(n)}
    for i, (a, b) in enumerate(edges):
        comb_adj[a].append((b, i))
        comb_adj[b].append((a, i))
    for j, (a, b) in enumerate(dummy):
        eid = m + j
        comb_adj[a].append((b, eid))
        comb_adj[b].append((a, eid))

    # Eulerian tour using Hierholzer (iterative).
    used = [False] * (m + len(dummy))
    stack = []
    tour = []  # list of vertex ids in order of the tour
    start = 0
    stack.append((start, 0))
    # Hierholzer iterative
    v = start
    tour_vertices = [v]
    # Use standard iterative Hierholzer
    stack2 = [(v, iter(list(comb_adj[v])))]
    while stack2:
        cur, it = stack2[-1]
        advanced = False
        for (w, eid) in it:
            if not used[eid]:
                used[eid] = True
                tour_vertices.append(w)
                stack2.append((w, iter(list(comb_adj[w]))))
                advanced = True
                break
        if not advanced:
            stack2.pop()

    # tour_vertices is the Eulerian trail (length = total_edges + 1)
    # Now split into paths at dummy edges.
    # Walk the tour; a path is a maximal segment where consecutive edges are real.
    total_edges = len(tour_vertices) - 1
    # map position (edge between vertex i and i+1) -> is it dummy?
    # We need to know which edge id was used at each step. Reconstruct by tracking.
    # Redo Hierholzer tracking edge ids.
    used = [False] * (m + len(dummy))
    stack2 = [(start, iter(list(comb_adj[start])))]
    tour_v = [start]
    tour_e = []  # edge id for each step
    while stack2:
        cur, it = stack2[-1]
        advanced = False
        for (w, eid) in it:
            if not used[eid]:
                used[eid] = True
                tour_v.append(w)
                tour_e.append(eid)
                stack2.append((w, iter(list(comb_adj[w]))))
                advanced = True
                break
        if not advanced:
            stack2.pop()

    # Now split: positions 0..len(tour_e)-1 correspond to edges.
    # A path is a maximal run of real edges (eid < m).
    paths = []
    i = 0
    L = len(tour_e)
    while i < L:
        if tour_e[i] < m:
            # start of a path
            seg = [tour_v[i]]
            j = i
            while j < L and tour_e[j] < m:
                seg.append(tour_v[j + 1])
                j += 1
            paths.append(seg)
            i = j
        else:
            i += 1

    # Validate: each path simple, edges are real and unused, cover all edges.
    seen_edges = set()
    for p in paths:
        if len(p) < 2:
            return _greedy_fallback(G, n, edges)
        for x in p:
            if x < 0 or x >= n:
                return _greedy_fallback(G, n, edges)
        # check simplicity
        if len(set(p)) != len(p):
            return _greedy_fallback(G, n, edges)
        for k in range(len(p) - 1):
            a, b = p[k], p[k + 1]
            e = (min(a, b), max(a, b))
            if e not in edge_index:
                return _greedy_fallback(G, n, edges)
            if e in seen_edges:
                return _greedy_fallback(G, n, edges)
            seen_edges.add(e)
    if len(seen_edges) != m:
        return _greedy_fallback(G, n, edges)
    return paths, True


def construct():
    results = []
    for spec in _specs:
        n, p, seed = spec
        edges = _instance(n, p, seed)
        G = (n, edges)
        if _time_left() < 3.0:
            # safety: return trivial decomposition
            results.append([list(e) for e in edges])
            continue
        paths, ok = _decompose(G, n, edges)
        if not ok:
            paths, ok = _greedy_fallback(G, n, edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
