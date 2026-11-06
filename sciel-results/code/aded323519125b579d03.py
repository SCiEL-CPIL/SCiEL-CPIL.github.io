"""Path decomposition via max-matching lower bound + greedy construction + local-search polish."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque


def construct():
    deadline = time.time() + 138.0  # safety margin before 150s kill
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(*spec)
        if not edges:
            results.append([])
            continue
        # Exact lower bound on # paths via max matching in edge-stub bipartite graph
        lb = _matching_lower_bound(edges)
        # Try to construct a decomposition achieving the bound with randomized restarts
        result = _solve(edges, lb, deadline)
        results.append(result)
    return results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _matching_lower_bound(edges):
    """
    Minimum number of edge-disjoint paths = |E| - |M|,
    where M is the maximum matching in the bipartite edge-stub graph.
    Left nodes: each edge e=(u,v) as 'starts at u' -> node (u, e_idx)
    Right nodes: each edge e=(u,v) as 'ends at v'   -> node (v, e_idx)
    Edge between left (u, e) and right (v, f) iff v==u and e != f.
    """
    n = len(edges)
    # Build adjacency: for each left node (u, e_idx), which right nodes (v, f_idx) are compatible
    # i.e., the edge f starts at the vertex where e ends.
    # e = (u, v) [u < v]. Left node = "e starts at u" = (u, e_idx).
    # Right node = "f ends at v" = (v, f_idx) where f = (v, w).
    # Actually let's think differently:
    # A path is a sequence of edges e1, e2, ..., ek where e_i ends at the vertex where e_{i+1} starts.
    # Edge e = (a, b) [a < b]. In a path, if we traverse e from a to b, then the next edge f must start at b.
    # If we traverse e from b to a, then the next edge f must start at a.
    # So for each undirected edge e, it has two "stubs": one at each endpoint.
    # Left side: stub (e, endpoint_a) meaning "e is entered at endpoint_a, exited at endpoint_b"
    # Right side: stub (f, endpoint) meaning "f is entered at endpoint"
    # Match left stub (e, a) to right stub (f, b) if a == b and e != f.
    # This means: after traversing e (exiting at b... wait, let me re-think.

    # Simpler formulation:
    # Each edge e has two orientations. In a path decomposition, each edge is used exactly once
    # in one orientation. A path is a sequence where consecutive edges share a vertex.
    # Think of it as: each edge e=(u,v) can be "entered" at u or at v.
    # Left node for edge e: "e is the next edge after some edge that ended at vertex X"
    # Right node for edge e: "e is the previous edge before some edge that starts at vertex Y"

    # Cleanest: bipartite graph with left = {e for e in E} (each edge as a node),
    # right = {e for e in E} (each edge as a node).
    # Left node e connects to right node f (e != f) if e and f share a vertex.
    # Then a matching M pairs edges that are consecutive in a path.
    # Number of paths = |E| - |M|.
    # BUT: we need the path to be simple (no repeated vertices), and the matching alone
    # doesn't guarantee that. However, for the LOWER BOUND, |E| - max_matching gives
    # the minimum number of trails (not necessarily simple paths). For simple paths,
    # the bound might be higher. But for these sparse random graphs, trails are usually
    # simple paths.

    # Build bipartite adjacency
    # left node i (edge i) -> right node j (edge j) if edges i and j share a vertex and i != j
    adj = [set() for _ in range(n)]
    for i in range(n):
        u, v = edges[i]
        for j in range(n):
            if i == j:
                continue
            a, b = edges[j]
            if v == a or v == b or u == a or u == b:
                adj[i].add(j)

    matching = _hopcroft_karp(adj, n)
    return n - matching


def _hopcroft_karp(adj, n):
    """Hopcroft-Karp maximum bipartite matching. adj[i] = set of right nodes adjacent to left node i."""
    pair_u = [-1] * n  # match for left node
    pair_v = [-1] * n  # match for right node
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
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
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching


def _solve(edges, lb, deadline):
    """
    Try to find a path decomposition with exactly lb paths using randomized greedy
    + local search. If we can't achieve lb, return the best we found.
    """
    n_edges = len(edges)
    if n_edges <= 1:
        return [[list(e)] for e in edges]

    best_paths = None
    best_count = n_edges

    # Phase 1: Randomized greedy with many restarts
    time_budget = deadline - time.time()
    num_restarts = min(200, max(20, int(time_budget * 5)))

    for attempt in range(num_restarts):
        if time.time() > deadline - 5:
            break
        rng = random.Random(attempt * 104729 + 7)
        paths = _greedy_decompose(edges, rng, deadline - 5)
        if paths is not None:
            # Local search: try to merge paths
            paths = _local_search(paths, edges, rng, deadline - 5)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                if count <= lb:
                    return best_paths

    # Phase 2: If we haven't hit the bound, try more aggressive local search
    if best_paths is not None and best_count > lb and time.time() < deadline - 5:
        rng = random.Random(99991)
        best_paths = _local_search(best_paths, edges, rng, deadline - 5)
        best_count = len(best_paths)

    if best_paths is None:
        return [list(e) for e in edges]
    return best_paths


def _greedy_decompose(edges, rng, deadline):
    """Greedy: repeatedly find a long simple path using remaining edges."""
    remaining = set(edges)
    if not remaining:
        return []

    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)

    degree = defaultdict(int)
    for u, v in remaining:
        degree[u] += 1
        degree[v] += 1

    paths = []

    while remaining:
        if time.time() > deadline:
            return None

        # Pick start: prefer odd-degree vertices, then high-degree
        odd_verts = [v for v in adj if adj[v] and degree.get(v, 0) % 2 == 1]
        if odd_verts:
            start = rng.choice(odd_verts)
        else:
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            weights = [max(1, degree.get(v, 0)) for v in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            start = candidates[-1]
            for v, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    start = v
                    break

        # Find long path from start
        path = _find_path(start, adj, degree, rng, remaining, deadline)

        if path is None or len(path) < 2:
            if adj[start]:
                nb = rng.choice(list(adj[start]))
                path = [start, nb]
            else:
                e = next(iter(remaining))
                path = [e[0], e[1]]

        # Remove path edges
        path_edges = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = tuple(sorted((u, v)))
            if e in remaining:
                path_edges.add(e)

        if not path_edges:
            break

        for e in path_edges:
            remaining.discard(e)
        for u, v in path_edges:
            adj[u].discard(v)
            adj[v].discard(u)
            degree[u] -= 1
            degree[v] -= 1

        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]

        paths.append(path)

    return paths


def _find_path(start, adj, degree, rng, remaining, deadline):
    """Find a long simple path using Warnsdorff + randomness."""
    path = [start]
    visited = {start}

    for _ in range(100):
        if time.time() > deadline:
            break
        cur = path[-1]
        neighbors = [nb for nb in adj[cur] if nb not in visited]
        if not neighbors:
            break

        def score(nb):
            unvisited = sum(1 for x in adj[nb] if x not in visited)
            return unvisited

        if rng.random() < 0.8:
            neighbors.sort(key=score)
            top_k = min(3, len(neighbors))
            choice = rng.choice(neighbors[:top_k])
        else:
            choice = rng.choice(neighbors)

        path.append(choice)
        visited.add(choice)

    return path if len(path) >= 2 else None


def _local_search(paths, edges, rng, deadline):
    """
    Try to reduce path count by:
    1. Merging two paths that share an endpoint (if result is simple).
    2. Re-routing: take an endpoint of one path and extend it into another path's edge.
    """
    edge_set = set(tuple(sorted(e)) for e in edges)
    paths = [list(p) for p in paths]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(paths)

        # Try all pairs for merging
        for i in range(n):
            if time.time() > deadline:
                return paths
            for j in range(n):
                if i == j:
                    continue
                merged = _try_merge(paths[i], paths[j], edge_set)
                if merged is not None:
                    paths.pop(j)
                    paths[i] = merged
                    improved = True
                    break
            if improved:
                break

        # Try re-routing: for path i, can we extend its end into an edge of path j?
        if not improved:
            for i in range(len(paths)):
                if time.time() > deadline:
                    return paths
                for j in range(len(paths)):
                    if i == j:
                        continue
                    new_i, new_j = _try_reroute(paths[i], paths[j], edge_set)
                    if new_i is not None:
                        paths[i] = new_i
                        paths[j] = new_j
                        improved = True
                        break
                if improved:
                    break

    return paths


def _try_merge(path_a, path_b, edge_set):
    """Try to concatenate two paths at a shared endpoint."""
    for a in (path_a, path_a[::-1]):
        for b in (path_b, path_b[::-1]):
            if a[-1] == b[0]:
                merged = a + b[1:]
                if _is_valid_path(merged, edge_set):
                    return merged
    return None


def _try_reroute(path_a, path_b, edge_set):
    """
    Try to take the last edge of path_b and attach it to the end of path_a,
    removing it from path_b. This is valid if:
    - path_a ends at vertex v, and path_b has an edge (v, w) at its end or start.
    - The resulting path_a' is simple.
    - path_b' (with that edge removed) is still a valid path (might become shorter).
    """
    # Try: path_a ends at v, path_b starts with edge (v, w)
    # Then new_a = path_a + [w], new_b = path_b[1:] (if len > 1, else discard)
    for a in (path_a, path_a[::-1]):
        for b in (path_b, path_b[::-1]):
            # a ends at a[-1], b starts with b[0]-b[1]
            if a[-1] == b[0] and len(b) >= 2:
                new_a = a + [b[1]]
                new_b = b[1:]
                if len(new_b) >= 2 and _is_valid_path(new_a, edge_set) and _is_valid_path(new_b, edge_set):
                    # Check edge-disjointness: new_a uses edge (a[-1], b[1]) which was in b
                    # This is fine since we're moving that edge from b to a
                    return new_a, new_b
            # a ends at a[-1], b ends with b[-2]-b[-1]
            if a[-1] == b[-1] and len(b) >= 2:
                new_a = a + [b[-2]]
                new_b = b[:-1]
                if len(new_b) >= 2 and _is_valid_path(new_a, edge_set) and _is_valid_path(new_b, edge_set):
                    return new_a, new_b
    return None, None


def _is_valid_path(path, edge_set):
    """Check path is simple and all edges exist in edge_set."""
    if len(path) < 2:
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        e = tuple(sorted((path[i], path[i + 1])))
        if e not in edge_set:
            return False
    return True

# EVOLVE-BLOCK-END
