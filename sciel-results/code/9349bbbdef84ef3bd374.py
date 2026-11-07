"""Rewritten program: exact branch-and-bound search for minimal triangle-free diameter-4 augmentation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# Precompute all candidate edges that are triangle-free w.r.t. the path
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

def _valid(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    d = _diameter(added)
    return d <= P['diameter']

def _exact_bnb(deadline):
    """Branch and bound to find minimum cardinality valid augmentation."""
    best_sol = None
    best_len = 10**9
    node_count = [0]
    
    # Sort candidates by "usefulness" - edges that reduce diameter most
    # Heuristic: prefer edges that connect far-apart vertices
    sorted_cands = sorted(_CANDIDATES, key=lambda e: -abs(e[0] - e[1]))
    
    def dfs(start_idx, current, adj_sets):
        nonlocal best_sol, best_len
        node_count[0] += 1
        
        if time.time() > deadline:
            return
        
        # Check if current solution is valid
        if len(current) >= best_len:
            return
        
        # Quick diameter check
        cur_diam = _diameter(current)
        if cur_diam <= P['diameter']:
            # Valid! Update best
            if len(current) < best_len:
                best_len = len(current)
                best_sol = list(current)
            return
        
        # Lower bound: need at least some more edges
        # If diameter is d, we need to reduce it to 4
        remaining = cur_diam - P['diameter']
        if len(current) + remaining >= best_len:
            return
        
        # Try adding each candidate from start_idx onwards
        for i in range(start_idx, len(sorted_cands)):
            if time.time() > deadline:
                return
            
            u, v = sorted_cands[i]
            
            # Check triangle-free constraint
            if adj_sets[u] & adj_sets[v]:
                continue
            
            # Add edge
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            current.append((u, v))
            
            dfs(i + 1, current, adj_sets)
            
            # Backtrack
            current.pop()
            adj_sets[u].discard(v)
            adj_sets[v].discard(u)
            
            if best_sol is not None and len(best_sol) <= 4:
                return
    
    # Initialize adjacency sets
    init_adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        init_adj[a].add(b)
        init_adj[b].add(a)
    
    dfs(0, [], init_adj)
    
    return best_sol, node_count[0]

def _construct_greedy(seed):
    rng = random.Random(seed)
    added = []
    added_set = set()
    cur = _diameter(added)
    while cur > P['diameter']:
        cands = []
        for (u, v) in _CANDIDATES:
            if (u, v) in added_set:
                continue
            # Check triangle-free
            adj = [set() for _ in range(_N)]
            for (a, b) in _BASE:
                adj[a].add(b)
                adj[b].add(a)
            for (a, b) in added:
                adj[a].add(b)
                adj[b].add(a)
            if adj[u] & adj[v]:
                continue
            cands.append((u, v))
        if not cands:
            return None
        best_edge = None
        best_d = 10**9
        if len(cands) > 200:
            cands = rng.sample(cands, 200)
        for (u, v) in cands:
            trial = added + [(u, v)]
            # Check triangle-free
            adj = [set() for _ in range(_N)]
            for (a, b) in _BASE:
                adj[a].add(b)
                adj[b].add(a)
            for (a, b) in trial:
                if adj[a] & adj[b]:
                    break
                adj[a].add(b)
                adj[b].add(a)
            else:
                d = _diameter(trial)
                if d < best_d:
                    best_d = d
                    best_edge = (u, v)
        if best_edge is None:
            return None
        added.append(best_edge)
        added_set.add(best_edge)
        cur = best_d
    return added

def _try_remove(added, seed):
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _local_search(added, seed, deadline):
    rng = random.Random(seed)
    cur = list(added)
    best = list(cur)
    best_len = len(cur)
    while time.time() < deadline:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                best = list(cur)
                best_len = len(cur)
                break
        if not improved:
            swapped = False
            for i in range(len(cur)):
                trial_remove = cur[:i] + cur[i+1:]
                for (u, v) in _CANDIDATES:
                    if (u, v) in set(trial_remove) or (u, v) in _BASE_SET:
                        continue
                    trial = trial_remove + [(u, v)]
                    if _valid(trial) and len(trial) < len(best):
                        cur = trial
                        best = list(cur)
                        best_len = len(cur)
                        swapped = True
                        break
                if swapped:
                    break
        if not improved and not swapped:
            break
    return best

def construct():
    deadline = time.time() + 55.0
    best = None
    
    # First try exact branch-and-bound
    exact_sol, nodes = _exact_bnb(deadline - 10)
    if exact_sol is not None:
        best = exact_sol
    else:
        # Fallback to heuristic
        for seed in range(20):
            if time.time() >= deadline:
                break
            g = _construct_greedy(seed)
            if g is not None and _valid(g):
                reduced = _try_remove(g, seed)
                if time.time() < deadline - 5:
                    reduced = _local_search(reduced, seed, deadline - 2)
                if best is None or len(reduced) < len(best):
                    best = reduced
            if best is not None and len(best) <= 5:
                break
    
    if best is None:
        return [[u, v] for u in range(_N) for v in range(u + 1, _N)
                if (u - v) % 2 and (u, v) not in _BASE_SET]
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
