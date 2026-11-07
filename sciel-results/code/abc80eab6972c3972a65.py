import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = set((min(u, v), max(u, v)) for u, v in PATH_EDGES)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]


def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_dist(adj, s):
    dist = [-1] * N
    dist[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _all_pairs_dist(adj):
    dist = []
    for s in range(N):
        dist.append(_bfs_dist(adj, s))
    return dist


def _max_dist(dist):
    mx = 0
    for d in dist:
        for x in d:
            if x > mx:
                mx = x
    return mx


def _is_connected(dist):
    for d in dist:
        for x in d:
            if x == -1:
                return False
    return True


def _diam_ok(adj):
    dist = _all_pairs_dist(adj)
    return _is_connected(dist) and _max_dist(dist) <= 4


def _tri_free(adj):
    for u in range(N):
        for v in adj[u]:
            if u < v:
                if set(adj[u]) & set(adj[v]):
                    return False
    return True


def _valid(added):
    adj = _build_adj(added)
    return _tri_free(adj) and _diam_ok(adj)


def _can_add(added_set, adj, u, v):
    if (min(u, v), max(u, v)) in added_set:
        return False
    if (min(u, v), max(u, v)) in PATH_SET:
        return False
    return not (set(adj[u]) & set(adj[v]))


def _reduce(added, deadline):
    added = list(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added)):
            trial = added[:i] + added[i + 1:]
            if _valid(trial):
                added = trial
                improved = True
                break
    return added


def _greedy_add(added, deadline):
    added = list(added)
    while time.time() < deadline:
        adj = _build_adj(added)
        if _diam_ok(adj):
            break
        best_edge = None
        best_diam = 999
        for (u, v) in ALL_PAIRS:
            if not _can_add(set(added), adj, u, v):
                continue
            trial = added + [(u, v)]
            adj2 = _build_adj(trial)
            if not _tri_free(adj2):
                continue
            dist = _all_pairs_dist(adj2)
            if not _is_connected(dist):
                continue
            d = _max_dist(dist)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
        if best_edge is None:
            break
        added.append(best_edge)
    return added


def _local_search(added, deadline):
    added = list(added)
    best = list(added)
    best_len = len(added)
    while time.time() < deadline:
        improved = False
        # Try removals
        for i in range(len(added)):
            trial = added[:i] + added[i + 1:]
            if _valid(trial):
                added = trial
                if len(added) < best_len:
                    best = list(added)
                    best_len = len(added)
                improved = True
                break
        if improved:
            continue
        # Try swaps
        for i in range(len(added)):
            if time.time() > deadline:
                break
            adj = _build_adj(added[:i] + added[i + 1:])
            for (u, v) in ALL_PAIRS:
                if not _can_add(set(added[:i] + added[i + 1:]), adj, u, v):
                    continue
                trial = added[:i] + added[i + 1:] + [(u, v)]
                if _valid(trial):
                    added = trial
                    if len(added) < best_len:
                        best = list(added)
                        best_len = len(added)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _two_hub_solution():
    """Try two-hub structure."""
    best = None
    for h1 in range(N):
        for h2 in range(N):
            if h2 <= h1:
                continue
            if abs(h1 - h2) < 2:
                continue
            # Check h1-h2 not creating triangle with path
            # h1 and h2 must not share a common neighbor on path
            n1 = set(PATH_EDGES[u] for u in [h1 - 1, h1 + 1] if 0 <= u < N)
            n2 = set(PATH_EDGES[u] for u in [h2 - 1, h2 + 1] if 0 <= u < N)
            # Actually, h1's path neighbors are h1-1 and h1+1
            n1 = {h1 - 1, h1 + 1} & set(range(N))
            n2 = {h2 - 1, h2 + 1} & set(range(N))
            if n1 & n2:
                continue  # h1 and h2 share a path neighbor, adding (h1,h2) creates triangle
            
            edges = []
            if abs(h1 - h2) >= 2:
                edges.append((min(h1, h2), max(h1, h2)))
            
            # Connect h1 to all vertices at distance >= 2 from h1, not adjacent to h1's path neighbors
            for v in range(N):
                if v == h1 or v == h2:
                    continue
                if abs(v - h1) < 2:
                    continue
                # Check triangle-free: v must not be adjacent to any of h1's current neighbors
                # h1's neighbors: path neighbors + h2 + other connected vertices
                # Simplify: just check with path
                if abs(v - (h1 - 1)) == 1 or abs(v - (h1 + 1)) == 1:
                    # v is adjacent to h1's path neighbor, triangle with (h1, path_neighbor, v)
                    # Only if path_neighbor is also connected to h1, which it is
                    continue
                edges.append((min(h1, v), max(h1, v)))
            
            # Connect h2 similarly
            for v in range(N):
                if v == h1 or v == h2:
                    continue
                if abs(v - h2) < 2:
                    continue
                if abs(v - (h2 - 1)) == 1 or abs(v - (h2 + 1)) == 1:
                    continue
                if (min(h2, v), max(h2, v)) not in edges:
                    edges.append((min(h2, v), max(h2, v)))
            
            if _valid(edges) and (best is None or len(edges) < len(best)):
                best = edges
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 85
    
    # Try multiple strategies
    candidates = []
    
    # Strategy 1: Single hub
    for hub in [11, 12, 10, 13, 7, 16]:
        edges = []
        for v in range(N):
            if v == hub:
                continue
            if abs(v - hub) < 2:
                continue
            # Check triangle-free: v not adjacent to hub's path neighbors
            if abs(v - (hub - 1)) == 1 or abs(v - (hub + 1)) == 1:
                continue
            edges.append((min(hub, v), max(hub, v)))
        if _valid(edges):
            candidates.append(edges)
    
    # Strategy 2: Two hubs
    two_hub = _two_hub_solution()
    if two_hub:
        candidates.append(two_hub)
    
    # Strategy 3: Greedy from empty
    greedy = _greedy_add([], deadline - 30)
    if _valid(greedy):
        candidates.append(greedy)
    
    # Strategy 4: Greedy from hub
    for hub in [11, 12]:
        seed = [(min(hub, v), max(hub, v)) for v in range(N)
                if v != hub and abs(v - hub) >= 2
                and abs(v - (hub - 1)) != 1 and abs(v - (hub + 1)) != 1]
        g = _greedy_add(seed, deadline - 30)
        if _valid(g):
            candidates.append(g)
    
    if not candidates:
        # Fallback
        edges = [(11, v) for v in [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]]
        candidates.append(edges)
    
    # Pick best and refine
    best = min(candidates, key=len)
    best = _reduce(best, deadline - 20)
    best = _local_search(best, deadline - 5)
    
    return [[int(u), int(v)] for (u, v) in best]
