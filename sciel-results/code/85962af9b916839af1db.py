"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import collections

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj):
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def _diameter(n, adj):
    maxd = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > maxd:
                        maxd = dist[v]
                    q.append(v)
        if -1 in dist:
            return float('inf')
    return maxd

def _valid(n, base, added):
    edges = base + added
    adj = _build_adj(n, edges)
    if not _is_triangle_free(n, adj):
        return None
    d = _diameter(n, adj)
    if d > 4:
        return None
    return d

def construct():
    n = P['n']
    base = [list(e) for e in P['edges']]
    base_set = set(tuple(sorted(e)) for e in base)
    deadline = time.time() + 58.0

    # Build a rich candidate pool of "safe" edges (length >= 5, no immediate triangle)
    pool = []
    for u in range(n):
        for v in range(u + 5, n):
            if (u, v) not in base_set:
                pool.append([u, v])

    # Start with a definitely-valid dense set: all step-5 and step-6 edges
    start = []
    for u in range(n - 5):
        start.append([u, u + 5])
    for u in range(n - 6):
        if [u, u + 6] not in start:
            start.append([u, u + 6])

    # Ensure start is valid; if not, add more until valid
    d = _valid(n, base, start)
    if d is None:
        # add all pool edges to guarantee validity
        start = list(pool)
        d = _valid(n, base, start)

    best = list(start)
    best_d = d
    if best_d is None:
        return []

    # Greedy removal
    def try_remove(cur):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(cur)):
                trial = cur[:idx] + cur[idx + 1:]
                d2 = _valid(n, base, trial)
                if d2 is not None:
                    cur = trial
                    improved = True
                    break
        return cur

    best = try_remove(best)

    # Local search: swap an edge for a shorter-count alternative is not applicable
    # (we already minimized count via removal). Try 1-swap to find smaller valid set
    # is already covered. Try adding then removing (2-opt style).
    if time.time() < deadline:
        for add_e in pool:
            if time.time() >= deadline:
                break
            if add_e in best:
                continue
            trial = best + [add_e]
            d2 = _valid(n, base, trial)
            if d2 is None:
                continue
            trial = try_remove(trial)
            if len(trial) < len(best):
                best = trial

    out = []
    for e in best:
        out.append([int(e[0]), int(e[1])])
    return out

# EVOLVE-BLOCK-END
