import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = set((min(u, v), max(u, v)) for u, v in PATH_EDGES)

# Precompute path distances
PATH_DIST = [[0] * N for _ in range(N)]
for u in range(N):
    for v in range(N):
        PATH_DIST[u][v] = abs(u - v)

# All candidate edges (non-path)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]


def _build(added):
    adj = [[] for _ in range(N)]
    mat = [[False] * N for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    for u, v in added:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    return adj, mat


def _bfs_all(adj):
    """Return list of BFS distance arrays from each vertex."""
    dists = []
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        dists.append(dist)
    return dists


def _diam_le4(dists):
    for dist in dists:
        for d in dist:
            if d > 4:
                return False
    return True


def _tri_free(adj, mat):
    for u in range(N):
        au = adj[u]
        for i in range(len(au)):
            a = au[i]
            for j in range(i + 1, len(au)):
                b = au[j]
                if mat[a][b]:
                    return False
    return True


def _valid_set(added):
    adj, mat = _build(added)
    if not _tri_free(adj, mat):
        return False
    dists = _bfs_all(adj)
    return _diam_le4(dists)


def _can_add(adj, mat, added_set, u, v):
    if u == v:
        return False
    k = (min(u, v), max(u, v))
    if k in PATH_SET or k in added_set:
        return False
    # Triangle-free: no common neighbor
    for w in adj[u]:
        if mat[w][v]:
            return False
    return True


def _hub_star(hub):
    """Build a star from hub to all vertices at path-distance >= 2."""
    edges = []
    for v in range(N):
        if v != hub and abs(v - hub) >= 2:
            edges.append((hub, v))
    return edges


def _reduce(added, deadline):
    """Try to remove each edge if the set remains valid."""
    added = set(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in list(added):
            trial = added - {e}
            if _valid_set(trial):
                added = trial
                improved = True
                break
    return added


def _local_search(added, deadline):
    """1-swap local search: replace one edge with another."""
    cur = set(added)
    best = set(cur)
    no_improve = 0
    while time.time() < deadline:
        cur = _reduce(cur, deadline)
        if len(cur) < len(best):
            best = set(cur)
            no_improve = 0
        adj, mat = _build(cur)
        cands = [p for p in ALL_PAIRS if p not in cur]
        random.shuffle(cands)
        improved = False
        e_list = list(cur)
        random.shuffle(e_list)
        for e in e_list:
            if time.time() > deadline:
                break
            for f in cands:
                if time.time() > deadline:
                    break
                trial = (cur - {e}) | {f}
                if _valid_set(trial):
                    trial = _reduce(trial, deadline)
                    if len(trial) < len(cur):
                        cur = trial
                        if len(cur) < len(best):
                            best = set(cur)
                        improved = True
                        break
            if improved:
                break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                # Kick: add a random edge then reduce
                if not cands:
                    break
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    f = random.choice(cands)
                    trial = cur | {f}
                    trial = _reduce(trial, deadline)
                    if len(trial) < len(cur):
                        cur = trial
                        if len(cur) < len(best):
                            best = set(cur)
                        no_improve = 0
                        break
                if no_improve >= 6:
                    break
    return best


def _greedy_complete(added, deadline):
    """Greedy: repeatedly add the edge that best reduces diameter."""
    added = set(added)
    while time.time() < deadline:
        adj, mat = _build(added)
        dists = _bfs_all(adj)
        if _diam_le4(dists):
            break
        # Find the pair with max distance
        max_d = 0
        max_pair = None
        for s in range(N):
            for t in range(N):
                if dists[s][t] > max_d:
                    max_d = dists[s][t]
                    max_pair = (s, t)
        if max_pair is None:
            break
        a, b = max_pair
        # Try to add edge (a,b) directly
        if _can_add(adj, mat, added, a, b):
            added.add((a, b))
            continue
        # Otherwise, find a vertex c such that adding (a,c) is valid and helps
        best_c = None
        best_gain = 0
        for c in range(N):
            if c == a:
                continue
            if not _can_add(adj, mat, added, a, c):
                continue
            # Estimate gain: new distance from b to a via c
            # Rough heuristic: prefer c that is close to b
            gain = PATH_DIST[b][c]
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c is not None:
            added.add((a, best_c))
        else:
            # Try any valid edge
            found = False
            for (u, v) in ALL_PAIRS:
                if _can_add(adj, mat, added, u, v):
                    added.add((u, v))
                    found = True
                    break
            if not found:
                break
    return added


def _solve_from(seed, deadline):
    added = _greedy_complete(set(seed), deadline)
    if not _valid_set(added):
        return None
    added = _reduce(added, deadline)
    return added


def _make_seeds():
    seeds = []
    # Hub stars for various hubs
    for hub in [11, 12, 7, 16, 8, 15, 5, 18]:
        seeds.append(_hub_star(hub))
    # Empty
    seeds.append([])
    # Small structured seeds
    for k in [5, 6, 7, 8]:
        added = []
        pos = 0
        while pos + k <= N - 1:
            added.append((pos, pos + k))
            pos += k
        if pos < N - 1:
            added.append((pos, N - 1))
        seeds.append(added)
    # Two-hub seeds
    for h1 in [6, 7, 8, 11, 12]:
        for h2 in [15, 16, 17, 18, 19]:
            if abs(h1 - h2) < 3:
                continue
            added = set()
            for v in range(N):
                d1 = abs(v - h1)
                d2 = abs(v - h2)
                if d1 >= 2 and d2 >= 2:
                    # Connect to closer hub
                    if d1 <= d2:
                        added.add((h1, v))
                    else:
                        added.add((h2, v))
            # Connect hubs
            if abs(h1 - h2) >= 2:
                added.add((h1, h2))
            seeds.append(list(added))
    return seeds


def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin: evaluator kills at 90s
    best = None

    # 1) Try all seeds
    seeds = _make_seeds()
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        sol = _solve_from(seed, deadline - 10)
        if sol is not None and _valid_set(sol):
            if best is None or len(sol) < len(best):
                best = sol

    # 2) Local search polish
    if best is not None and time.time() < deadline - 3:
        polished = _local_search(best, deadline - 3)
        if _valid_set(polished) and len(polished) < len(best):
            best = polished

    # 3) Multiple random restarts with local search
    if time.time() < deadline - 3:
        for restart in range(30):
            if time.time() > deadline - 3:
                break
            # Random seed: pick random subset of ALL_PAIRS
            random.seed(restart * 137 + 42)
            k = random.randint(5, 12)
            random.seed(restart * 137 + 42)
            seed = random.sample(ALL_PAIRS, min(k, len(ALL_PAIRS)))
            sol = _solve_from(seed, deadline - 5)
            if sol is not None and _valid_set(sol):
                sol = _local_search(sol, deadline - 3)
                if _valid_set(sol) and (best is None or len(sol) < len(best)):
                    best = sol

    if best is None:
        # Fallback: hub 12 star
        best = set(_hub_star(12))
        if not _valid_set(best):
            best = set(_hub_star(11))

    return [[int(u), int(v)] for (u, v) in best]
