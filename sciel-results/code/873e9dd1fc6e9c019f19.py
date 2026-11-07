import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# All candidate edges: not base, not distance-2 (those create triangles with path)
_CANDIDATES = []
for u in range(_N):
    for v in range(u + 1, _N):
        if (u, v) in _BASE_SET:
            continue
        if abs(u - v) == 2:
            continue
        _CANDIDATES.append((u, v))

def _build_adj(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _diameter(added):
    adj = _build_adj(added)
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        if -1 in dist:
            return 10**9
        m = max(dist)
        if m > best:
            best = m
    return best

def _is_triangle_free_set(added):
    # Incremental triangle-free check
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    return True

def _valid(added):
    if not _is_triangle_free_set(added):
        return False
    d = _diameter(added)
    return d <= P['diameter']

def _shares_neighbor(adj, u, v):
    return bool(adj[u] & adj[v])

def _star_search(center, budget_edges, deadline):
    """For a given center, greedily pick edges from center to reduce diameter.
    The star from center is inherently triangle-free (no two leaves share center's
    neighbor beyond the path, as long as leaves aren't distance 2 apart on path)."""
    # Build adjacency with just base
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    
    # Potential star edges from center: not base, not distance 2
    star_cands = []
    for v in range(_N):
        if v == center:
            continue
        if (min(center, v), max(center, v)) in _BASE_SET:
            continue
        if abs(center - v) == 2:
            continue
        star_cands.append(v)
    
    added = []
    added_set = set()
    cur_diam = _diameter(added)
    
    while cur_diam > P['diameter'] and len(added) < budget_edges:
        if time.time() >= deadline:
            return None
        best_edge = None
        best_d = 10**9
        for v in star_cands:
            if v in added_set:
                continue
            # Check triangle-free: center-v must not share neighbor with existing
            if _shares_neighbor(adj, center, v):
                continue
            trial = added + [(center, v)]
            d = _diameter(trial)
            if d < best_d:
                best_d = d
                best_edge = v
        if best_edge is None:
            return None
        added.append((center, best_edge))
        added_set.add(best_edge)
        adj[center].add(best_edge)
        adj[best_edge].add(center)
        cur_diam = best_d
    
    if cur_diam <= P['diameter']:
        return added
    return None

def _try_remove(added, seed, deadline):
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(cur)))
        rng.shuffle(indices)
        for i in indices:
            if time.time() >= deadline:
                return cur
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _local_search(added, seed, deadline):
    """Try swapping edges to reduce count or find better config."""
    rng = random.Random(seed)
    cur = list(added)
    cur_valid = _valid(cur)
    if not cur_valid:
        return cur
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(cur)))
        rng.shuffle(indices)
        for i in indices:
            if time.time() >= deadline:
                return cur
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s kill
    best = None
    
    # Try star-based search from each center
    for center in range(_N):
        if time.time() >= deadline:
            break
        for budget in range(5, 11):
            if time.time() >= deadline:
                break
            sol = _star_search(center, budget, deadline)
            if sol is not None and _valid(sol):
                reduced = _try_remove(sol, center * 100 + budget, deadline)
                if best is None or len(reduced) < len(best):
                    best = reduced
                    if len(best) <= 5:
                        return [[int(u), int(v)] for (u, v) in best]
    
    # If no star solution, fall back to general greedy
    if best is None:
        for seed in range(20):
            if time.time() >= deadline:
                break
            added = []
            added_set = set()
            cur_diam = _diameter(added)
            while cur_diam > P['diameter']:
                if time.time() >= deadline:
                    break
                adj = _build_adj(added)
                cands = []
                for (u, v) in _CANDIDATES:
                    if (u, v) in added_set:
                        continue
                    if _shares_neighbor(adj, u, v):
                        continue
                    cands.append((u, v))
                if not cands:
                    break
                best_edge = None
                best_d = 10**9
                rng = random.Random(seed)
                rng.shuffle(cands)
                for (u, v) in cands:
                    trial = added + [(u, v)]
                    d = _diameter(trial)
                    if d < best_d:
                        best_d = d
                        best_edge = (u, v)
                if best_edge is None:
                    break
                added.append(best_edge)
                added_set.add(best_edge)
                cur_diam = best_d
            if _valid(added):
                reduced = _try_remove(added, seed, deadline)
                if best is None or len(reduced) < len(best):
                    best = reduced
                    if len(best) <= 5:
                        break
    
    if best is None:
        return []
    
    return [[int(u), int(v)] for (u, v) in best]
