"""Fresh design: bipartite (even/odd) shortcut augmentation, greedy + local search,
with an explicit wall-clock deadline guard."""
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14],
                        [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
                        [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = P['edges']
D = 4
DEADLINE = time.time() + 80.0  # safety margin before the 90s wall-clock kill


def _dist_matrix(adj):
    """BFS from every vertex; returns dist[i][j] or INF if unreachable."""
    INF = 10**9
    dist = [[INF] * N for _ in range(N)]
    for s in range(N):
        dq = [s]
        dist[s][s] = 0
        while dq:
            nq = []
            for u in dq:
                du = dist[s][u] + 1
                for v in adj[u]:
                    if dist[s][v] > du:
                        dist[s][v] = du
                        nq.append(v)
            dq = nq
    return dist


def _is_triangle_free(adj):
    for u in range(N):
        su = set(adj[u])
        for v in adj[u]:
            if v <= u:
                continue
            if su & set(adj[v]):
                return False
    return True


def _valid(added):
    adj = [[] for _ in range(N)]
    for u, v in BASE:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    if not _is_triangle_free(adj):
        return False
    dist = _dist_matrix(adj)
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i][j] > D:
                return False
    return True


def _pair_status(added):
    """Return (dist, set of (i,j) with i<j and dist>D)."""
    adj = [[] for _ in range(N)]
    for u, v in BASE:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    dist = _dist_matrix(adj)
    bad = set()
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i][j] > D:
                bad.add((i, j))
    return dist, bad


def _fixes(e, bad):
    """Number of currently-failing pairs that a single edge e (in the *current*
    graph) would resolve. Uses the precomputed dist matrix."""
    return None  # placeholder, replaced below


def _greedy():
    """Greedy: repeatedly add the cross-parity edge that resolves the most
    currently-failing pairs (tie-break: largest max pair-distance it touches),
    skipping any edge that would create a triangle."""
    adj = [[] for _ in range(N)]
    for u, v in BASE:
        adj[u].append(v)
        adj[v].append(u)
    added = []
    used = set()
    # cross-parity candidate pool
    cands = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u - v) % 2 and (u, v) not in {(a, b) for a, b in BASE}:
                cands.append((u, v))

    dist, bad = _pair_status(added)
    while bad and time.time() < DEADLINE:
        best = None
        best_key = None
        for (u, v) in cands:
            if (u, v) in used:
                continue
            # triangle check: no common neighbour of u and v
            if set(adj[u]) & set(adj[v]):
                continue
            # resolve count: pairs (i,j) in bad where new edge reduces dist to <= D
            # new distance for pair (i,j) = min(dist[i][j], dist[i][u]+1+dist[v][j],
            #                                   dist[i][v]+1+dist[u][j])
            cnt = 0
            mx = -1
            for (i, j) in bad:
                nd = dist[i][j]
                a = dist[i][u] + 1 + dist[v][j]
                b = dist[i][v] + 1 + dist[u][j]
                if a < nd:
                    nd = a
                if b < nd:
                    nd = b
                if nd <= D:
                    cnt += 1
                    if dist[i][j] > mx:
                        mx = dist[i][j]
            if cnt <= 0:
                continue
            key = (cnt, mx)
            if best_key is None or key > best_key:
                best_key = key
                best = (u, v)
        if best is None:
            break
        u, v = best
        used.add((u, v))
        added.append([u, v])
        adj[u].append(v)
        adj[v].append(u)
        dist, bad = _pair_status(added)
    return added


def _local_search(added, max_time):
    """Try to reduce edge count via swaps / removals, keeping validity."""
    cur = [tuple(e) for e in added]
    best = list(cur)
    t0 = time.time()
    cands = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u - v) % 2 and (u, v) not in {(a, b) for a, b in BASE}:
                cands.append((u, v))
    # Removal pass
    changed = True
    while changed and time.time() < max_time:
        changed = False
        for idx in range(len(cur)):
            cand = cur[:idx] + cur[idx + 1:]
            if _valid(cand):
                cur = cand
                changed = True
                best = list(cur)
                break
    # Swap pass
    while time.time() < max_time:
        improved = False
        for idx in range(len(cur)):
            for c in cands:
                if c in cur:
                    continue
                cand = cur[:idx] + cur[idx + 1:] + [c]
                if len(cand) < len(cur) and _valid(cand):
                    cur = cand
                    best = list(cur)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def construct():
    deadline = DEADLINE
    sol = _greedy()
    if not _valid(sol):
        # fallback: keep adding any non-triangle cross-parity edge until valid
        adj = [[] for _ in range(N)]
        for u, v in BASE:
            adj[u].append(v)
            adj[v].append(u)
        used = set()
        cands = []
        for u in range(N):
            for v in range(u + 1, N):
                if (u - v) % 2 and (u, v) not in {(a, b) for a, b in BASE}:
                    cands.append((u, v))
        while not _valid(sol) and time.time() < deadline:
            for (u, v) in cands:
                if (u, v) in used:
                    continue
                if set(adj[u]) & set(adj[v]):
                    continue
                used.add((u, v))
                sol.append([u, v])
                adj[u].append(v)
                adj[v].append(u)
                break
            else:
                break
    if _valid(sol) and time.time() < deadline - 15:
        reduced = _local_search(sol, deadline - 15)
        if _valid(reduced) and len(reduced) <= len(sol):
            sol = reduced
    # enforce exact types: list of [int, int]
    out = [[int(u), int(v)] for (u, v) in sol]
    return out
