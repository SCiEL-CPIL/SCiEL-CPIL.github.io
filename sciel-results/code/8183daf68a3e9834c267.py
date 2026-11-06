"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _max_bipartite_matching(n, left_adj):
    """Maximum bipartite matching using Hopcroft-Karp for efficiency.
    left_adj[u] = list of v (right nodes) adjacent to left node u.
    Returns match_right[v] = u or -1.
    """
    from collections import deque
    INF = float('inf')
    matchL = [-1] * n  # matchL[u] = v
    matchR = [-1] * n  # matchR[v] = u
    dist = [0] * n

    def bfs():
        q = deque()
        for u in range(n):
            if matchL[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in left_adj[u]:
                mu = matchR[v]
                if mu == -1:
                    found = True
                elif dist[mu] == INF:
                    dist[mu] = dist[u] + 1
                    q.append(mu)
        return found

    def dfs(u):
        for v in left_adj[u]:
            mu = matchR[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if matchL[u] == -1:
                if dfs(u):
                    matching += 1
    return matchR, matching

def _solve_graph(n, p, seed, deadline=None):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []

    edge_set = set()
    for (u, v) in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Build bipartite adjacency: left[u] can match right[v] iff (u,v) is an edge
    left_adj = [[] for _ in range(n)]
    for (u, v) in edges:
        left_adj[u].append(v)
        left_adj[v].append(u)

    matchR, matching_size = _max_bipartite_matching(n, left_adj)

    # Build directed structure from matching
    # If left[u] matched to right[v], then directed edge u -> v
    succ = [-1] * n  # succ[u] = v
    pred = [-1] * n  # pred[v] = u
    matched_edge_set = set()

    for v in range(n):
        u = matchR[v]
        if u != -1:
            succ[u] = v
            pred[v] = u
            matched_edge_set.add((min(u, v), max(u, v)))

    # Extract paths and cycles
    all_paths = []
    visited = [False] * n

    # Paths: start at vertices with pred == -1 and succ != -1
    for v in range(n):
        if pred[v] == -1 and succ[v] != -1:
            path = [v]
            current = v
            while succ[current] != -1:
                current = succ[current]
                path.append(current)
                visited[current] = True
            visited[v] = True
            all_paths.append(path)

    # Cycles: vertices with pred != -1 and succ != -1 not yet visited
    for v in range(n):
        if pred[v] != -1 and succ[v] != -1 and not visited[v]:
            # Follow to form a cycle
            cycle = [v]
            seen = {v}
            current = v
            while True:
                current = succ[current]
                if current in seen:
                    if current == v:
                        all_paths.append(cycle)
                    break
                cycle.append(current)
                seen.add(current)
            for node in cycle:
                visited[node] = True

    # Add single-edge paths for unmatched edges
    for (u, v) in edges:
        e = (min(u, v), max(u, v))
        if e not in matched_edge_set:
            all_paths.append([u, v])

    # Validate
    covered = set()
    valid = True
    for path in all_paths:
        if len(path) < 2:
            continue
        if len(set(path)) != len(path):
            valid = False
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (min(a, b), max(a, b))
            if e in covered or e not in edge_set:
                valid = False
                break
            covered.add(e)
        if not valid:
            break

    if not valid or covered != edge_set:
        # Fallback: each edge as its own path
        return [[u, v] for (u, v) in edges]

    return all_paths

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            paths = _solve_graph(*spec)
            results.append(paths)

    return results
